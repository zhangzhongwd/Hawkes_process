import pandas as pd  # 导入pandas库，用于数据处理
import numpy as np  # 导入numpy库，用于数值计算
import pandas_ta as ta  # 导入pandas_ta库，用于技术分析指标
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import scipy  # 导入scipy库，提供科学计算功能


def plot_two_axes(series1, *ex_series):
    """
    绘制双坐标轴图表
    :param series1: 主坐标轴数据序列（通常为价格）
    :param ex_series: 副坐标轴数据序列（可多个，通常为指标）
    """
    plt.style.use('dark_background')  # 使用黑色背景主题
    ax = series1.plot(color='green')  # 主序列（价格）用绿色
    ax2 = ax.twinx()  # 创建副坐标轴
    for i, series in enumerate(ex_series):
        series.plot(ax=ax2, alpha=0.5)  # 其他序列（指标）用半透明线
    #plt.show()  # 显示图表（注释掉以避免自动显示）

def hawkes_process(data: pd.Series, kappa: float):
    """
    Hawkes过程滤波器实现
    :param data: 输入数据序列（标准化波动率）
    :param kappa: 衰减系数（越大表示记忆效应越短）
    :return: 处理后的波动率序列
    """
    assert(kappa > 0.0)  # 确保kappa为正值
    alpha = np.exp(-kappa)  # 计算指数衰减系数
    arr = data.to_numpy()  # 将数据序列转换为numpy数组
    output = np.zeros(len(data))  # 初始化输出数组
    output[:] = np.nan  # 初始化NaN数组
    for i in range(1, len(data)):
        if np.isnan(output[i - 1]):  # 处理初始值
            output[i] = arr[i]  # 如果前一个值为NaN，直接赋值
        else:
            # 核心公式：当前值 = 前值衰减 + 新值
            output[i] = output[i - 1] * alpha + arr[i]  # 计算当前值
    return pd.Series(output, index=data.index) * kappa  # 最终乘以kappa缩放并返回

def vol_signal(close: pd.Series, vol_hawkes: pd.Series, lookback:int):
    """
    波动率交易信号生成器
    :param close: 价格序列
    :param vol_hawkes: 处理后的波动率序列
    :param lookback: 滚动窗口长度（用于计算分位数阈值）
    :return: 交易信号数组（-1,0,1）
    """
    signal = np.zeros(len(close))  # 初始化信号数组
    q05 = vol_hawkes.rolling(lookback).quantile(0.05)  # 5%分位数（低波动阈值）
    q95 = vol_hawkes.rolling(lookback).quantile(0.95)  # 95%分位数（高波动阈值）
    
    last_below = -1  # 记录最后一次低于低阈值的位置
    curr_sig = 0     # 当前信号（-1,0,1）

    for i in range(len(signal)):
        if vol_hawkes.iloc[i] < q05.iloc[i]:  # 当波动率低于低阈值
            last_below = i  # 更新最后低波动位置
            curr_sig = 0    # 重置信号
        
        # 当波动率突破高阈值，且前一期未突破，且有历史低点
        if vol_hawkes.iloc[i] > q95.iloc[i] \
           and vol_hawkes.iloc[i - 1] <= q95.iloc[i - 1] \
           and last_below > 0 :
            
            change = close.iloc[i] - close.iloc[last_below]  # 计算价格变化
            curr_sig = 1 if change > 0.0 else -1  # 根据价格变化方向确定信号
        signal[i] = curr_sig  # 更新信号数组

    return signal  # 返回交易信号数组

def get_trades_from_signal(data: pd.DataFrame, signal: np.array):
    """
    从信号中获取交易的进出场时间
    :param data: 输入数据（包含价格信息）
    :param signal: 交易信号数组
    :return: 做多和做空交易的DataFrame
    """
    long_trades = []  # 存储做多交易
    short_trades = []  # 存储做空交易

    close_arr = data['close'].to_numpy()  # 将收盘价转换为numpy数组
    last_sig = 0.0  # 上一个信号值
    open_trade = None  # 当前打开的交易
    idx = data.index  # 数据的索引（时间）

    for i in range(len(data)):
        if signal[i] == 1.0 and last_sig != 1.0:  # 做多入场
            if open_trade is not None:  # 如果有未关闭的交易，关闭它
                open_trade[2] = idx[i]  # 设置退出时间
                open_trade[3] = close_arr[i]  # 设置退出价格
                short_trades.append(open_trade)  # 添加到做空交易列表

            open_trade = [idx[i], close_arr[i], -1, np.nan]  # 开始新的做多交易
        if signal[i] == -1.0 and last_sig != -1.0:  # 做空入场
            if open_trade is not None:  # 如果有未关闭的交易，关闭它
                open_trade[2] = idx[i]  # 设置退出时间
                open_trade[3] = close_arr[i]  # 设置退出价格
                long_trades.append(open_trade)  # 添加到做多交易列表

            open_trade = [idx[i], close_arr[i], -1, np.nan]  # 开始新的做空交易
        
        if signal[i] == 0.0 and last_sig == -1.0:  # 做空出场
            open_trade[2] = idx[i]  # 设置退出时间
            open_trade[3] = close_arr[i]  # 设置退出价格
            short_trades.append(open_trade)  # 添加到做空交易列表
            open_trade = None  # 重置当前交易

        if signal[i] == 0.0 and last_sig == 1.0:  # 做多出场
            open_trade[2] = idx[i]  # 设置退出时间
            open_trade[3] = close_arr[i]  # 设置退出价格
            long_trades.append(open_trade)  # 添加到做多交易列表
            open_trade = None  # 重置当前交易

        last_sig = signal[i]  # 更新上一个信号值

    # 将交易列表转换为DataFrame，并计算每笔交易的收益百分比
    long_trades = pd.DataFrame(long_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])
    short_trades = pd.DataFrame(short_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])

    long_trades['percent'] = (long_trades['exit_price'] - long_trades['entry_price']) / long_trades['entry_price']  # 计算做多交易收益百分比
    short_trades['percent'] = -1 * (short_trades['exit_price'] - short_trades['entry_price']) / short_trades['entry_price']  # 计算做空交易收益百分比
    long_trades = long_trades.set_index('entry_time')  # 设置做多交易的索引为入场时间
    short_trades = short_trades.set_index('entry_time')  # 设置做空交易的索引为入场时间
    return long_trades, short_trades  # 返回做多和做空交易

# 读取数据
data = pd.read_csv('VolatilityHawkes-main/BTCUSDT3600.csv')  # 从CSV文件读取数据
data['date'] = data['date'].astype('datetime64[s]')  # 将日期列转换为datetime格式
data = data.set_index('date')  # 将日期列设置为索引

# 归一化波动率
norm_lookback = 336  # 设置滚动窗口长度
data['atr'] = ta.atr(np.log(data['high']), np.log(data['low']), np.log(data['close']), norm_lookback)  # 计算平均真实波动范围（ATR）
data['norm_range'] = (np.log(data['high']) - np.log(data['low'])) / data['atr']  # 计算标准化范围

# 计算Hawkes过程波动率和交易信号
data['v_hawk'] = hawkes_process(data['norm_range'], 0.1)  # 计算Hawkes过程波动率
data['sig'] = vol_signal(data['close'], data['v_hawk'], 168)  # 生成交易信号

# 计算信号收益
data['next_return'] = np.log(data['close']).diff().shift(-1)  # 计算下一个收益
data['signal_return'] = data['sig'] * data['next_return']  # 计算信号收益
win_returns = data[data['signal_return'] > 0]['signal_return'].sum()  # 计算盈利收益
lose_returns = data[data['signal_return'] < 0]['signal_return'].abs().sum()  # 计算亏损收益
signal_pf = win_returns / lose_returns  # 计算盈利因子
plt.style.use('dark_background')  # 设置绘图风格
data['signal_return'].cumsum().plot()  # 绘制累计收益图

# 获取交易并计算统计数据
long_trades, short_trades = get_trades_from_signal(data, data['sig'].to_numpy())  # 从信号中获取交易
long_win_rate = len(long_trades[long_trades['percent'] > 0]) / len(long_trades)  # 计算做多胜率
short_win_rate = len(short_trades[short_trades['percent'] > 0]) / len(short_trades)  # 计算做空胜率
long_average = long_trades['percent'].mean()  # 计算做多平均收益
short_average = short_trades['percent'].mean()  # 计算做空平均收益
time_in_market = len(data[data['sig'] != 0.0]) / len(data)  # 计算市场持有时间比例

# 打印统计结果
print("Profit Factor", signal_pf)  # 打印盈利因子
print("Long Win Rate", long_win_rate)  # 打印做多胜率
print("Long Average", long_average)  # 打印做多平均收益
print("Short Win Rate", short_win_rate)  # 打印做空胜率
print("Short Average", short_average)  # 打印做空平均收益
print("Time In Market", time_in_market)  # 打印市场持有时间比例

# # 生成热力图
# kappa_vals = [0.5, 0.25, 0.1, 0.05, 0.01]  # 设置Hawkes过程的kappa值
# lookback_vals = [24, 48, 96, 168, 336]  # 设置滚动窗口长度
# pf_df = pd.DataFrame(index=lookback_vals, columns=kappa_vals)  # 创建空的DataFrame用于存储盈利因子

# for lb in lookback_vals:  # 遍历每个滚动窗口长度
#     for k in kappa_vals:  # 遍历每个kappa值
#         data['v_hawk'] = hawkes_process(data['norm_range'], k)  # 计算Hawkes过程波动率
#         data['sig'] = vol_signal(data['close'], data['v_hawk'], lb)  # 生成交易信号

#         data['next_return'] = np.log(data['close']).diff().shift(-1)  # 计算下一个收益
#         data['signal_return'] = data['sig'] * data['next_return']  # 计算信号收益
#         win_returns = data[data['signal_return'] > 0]['signal_return'].sum()  # 计算盈利收益
#         lose_returns = data[data['signal_return'] < 0]['signal_return'].abs().sum()  # 计算亏损收益
#         signal_pf = win_returns / lose_returns  # 计算盈利因子

#         pf_df.loc[lb, k] = float(signal_pf)  # 将盈利因子存入DataFrame
    
# plt.style.use('dark_background')  # 设置绘图风格
# import seaborn as sns  # 导入seaborn库，用于绘制热力图
# pf_df = pf_df.astype(float)  # 将DataFrame转换为浮点数类型
# sns.heatmap(pf_df, annot=True, fmt='f')  # 绘制热力图
# plt.xlabel('Hawkes Kappa')  # 设置x轴标签
# plt.ylabel('Threshold Lookback')  # 设置y轴标签
# plt.show()  # 显示图表






# 启用交互模式
plt.ion()  # 启用交互模式

# ... existing code ...

# 绘制BTC价格和Hawkes过程波动率
plt.figure(figsize=(14, 7))  # 设置图形大小

# 绘制BTC价格
ax1 = plt.gca()  # 获取当前坐标轴
ax1.plot(data.index, data['close'], label='BTC Price', color='blue')  # 绘制BTC价格
ax1.set_xlabel('Time')  # 设置x轴标签
ax1.set_ylabel('BTC Price', color='blue')  # 设置左侧y轴标签
ax1.tick_params(axis='y', labelcolor='blue')  # 设置左侧y轴刻度颜色

# 创建共享x轴的第二个y轴
ax2 = ax1.twinx()  # 创建副坐标轴
ax2.plot(data.index, data['v_hawk'], label='Hawkes Process Volatility', color='orange')  # 绘制Hawkes过程波动率
ax2.plot(data.index, data['norm_range'], label='Normalized Range', color='green', alpha=0.5)  # 新增标准化波动率线
ax2.set_ylabel('Volatility', color='black')  # 设置右侧通用y轴标签
ax2.tick_params(axis='y', labelcolor='black')  # 设置右侧y轴刻度颜色

plt.title('BTC Price and Volatility Indicators')  # 设置标题
ax1.legend(loc='upper left')  # BTC价格图例
ax2.legend(loc='upper right')  # 波动率指标图例

# 显示图表（阻塞模式）
plt.show(block=True)  # 程序会在此处阻塞，直到手动关闭窗口

# ... existing code ...
