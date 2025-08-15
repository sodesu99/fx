# features.py

import numpy as np
import pandas as pd
from scipy.stats import rankdata

def calculate_rci(series, period=9):
    """计算 Rank Correlation Index"""
    n = len(series)
    if n < period:
        return np.nan
    recent = series[-period:]
    ranks = rankdata(recent)
    # 计算与时间序列的皮尔逊相关
    time_ranks = np.arange(1, period + 1)
    corr = np.corrcoef(time_ranks, ranks)[0, 1]
    return corr * 100  # 百分比形式

def add_rci_features(df, periods=[9, 14, 21]):
    """添加多个RCI列"""
    for p in periods:
        df[f'RCI{p}'] = np.nan
        for i in range(p, len(df)):
            df.loc[df.index[i], f'RCI{p}'] = calculate_rci(df['close'].iloc[i-p:i], p)
    return df

def bayesian_signal_probability(df, signal_col, future_return_col, threshold=0.0005, prior=None):
    """
    计算 P(上涨 | 信号) 使用贝叶斯
    """
    # 先验：默认市场中性
    P_B = prior or 0.5  # P(上涨)

    # 似然：P(信号 | 上涨)
    is_up = df[future_return_col] > threshold
    signals = df[signal_col]
    P_A_given_B = signals[is_up].mean() / is_up.mean() if is_up.mean() > 0 else 0.5

    # P(信号)
    P_A = signals.mean()

    # 后验
    if P_A > 0:
        P_B_given_A = (P_A_given_B * P_B) / P_A
    else:
        P_B_given_A = 0.5

    return min(max(P_B_given_A, 0), 1)  # clamp to [0,1]



def generate_features(df, lookback=50):
    """生成完整特征向量"""
    df = df.copy()
    df = add_rci_features(df, [9,14,21])

    # 添加其他技术特征
    df['Return_1h'] = df['close'].pct_change()
    df['Volatility'] = df['Return_1h'].rolling(20).std()
    df['MA50_Close_Ratio'] = df['close'] / df['close'].rolling(50).mean()

    # 创建信号：3 RCI共振（都 < -80）
    df['signal_rci_combo'] = (df['RCI9'] < -80) & (df['RCI14'] < -70) & (df['RCI21'] < -60)

    # 提前计算未来收益（用于贝叶斯）
    df['future_return_5h'] = (df['close'].shift(-5) - df['close']) / df['close']

    # 滚动窗口计算贝叶斯后验概率（使用前N天数据）
    window_size = 100
    bayes_probs = [np.nan] * window_size

    for i in range(window_size, len(df)):
        window = df.iloc[i-window_size:i]
        prob = bayesian_signal_probability(
            window,
            signal_col='signal_rci_combo',
            future_return_col='future_return_5h',
            threshold=0.0005
        )
        bayes_probs.append(prob)

    df['Bayesian_Prob'] = bayes_probs

    # 填充 NaN
    df.fillna(method='ffill', inplace=True)
    df.fillna(0.5, inplace=True)

    return df

# 使用示例
# df = pd.read_csv('EURUSD_1H_2023.csv', index_col='timestamp', parse_dates=True)
# df = generate_features(df)
# print(df[['close', 'RCI9', 'RCI14', 'RCI21', 'Bayesian_Prob']].tail())