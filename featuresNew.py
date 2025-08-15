# features.py
import numpy as np
import pandas as pd

def calculate_rci(series, period=9):
    n = len(series)
    if n < period:
        return np.nan
    recent = series[-period:]
    ranks = np.array([sorted(recent).index(x) + 1 for x in recent])
    time_ranks = np.arange(1, period + 1)
    corr = np.corrcoef(time_ranks, ranks)[0, 1]
    return corr * 100

def add_features(df0):
    df = df0.copy()

    df['RCI9'] = df['close'].rolling(9).apply(calculate_rci, raw=False)
    df['RCI14'] = df['close'].rolling(14).apply(calculate_rci, raw=False)
    df['RCI21'] = df['close'].rolling(21).apply(calculate_rci, raw=False)

    df['Volatility'] = df['close'].pct_change().rolling(20).std()
    df['MA50_Ratio'] = df['close'] / df['close'].rolling(50).mean()

    # 贝叶斯先验信号：假设我们有历史统计
    df['Bayesian_Prob'] = 0.5  # 可替换为真实后验（见前文）
    df['Bayesian_Prob'] = df['Bayesian_Prob'].fillna(0.5)




    # rolling window Size
    window = 50

    # 显式转为浮点型
    for col in ['close', 'high', 'low', 'volume']:
         df[col] = df[col].astype(float)
    
    # 计算高低价范围
    df['High_Low_Range'] = (df['high'] - df['low']) / df['close']
    # 归一化收盘价
    df['close_norm'] = (df['close'] - df['close'].rolling(window).min()) / (df['close'].rolling(window).max() - df['close'].rolling(window).min())

    # 归一化成交量
    df['volume_norm'] = (df['volume'] - df['volume'].rolling(window).min()) / (df['volume'].rolling(window).max() - df['volume'].rolling(window).min())

    # 删除原始列
    # df = df.drop(columns=['close', 'high', 'low', 'volume'])
    
    

    df.dropna(inplace=True)
    return df