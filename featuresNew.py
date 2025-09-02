# featuresNew.py - 改进版
import numpy as np
import pandas as pd
from typing import Optional

def calculate_rci_vectorized(series: pd.Series, period: int = 9) -> pd.Series:
    """
    ✅ 优化版：向量化RCI计算，提升效率
    """
    def rci_single(x):
        if len(x) < period or x.isna().any():
            return np.nan
        ranks = x.rank(method='average')
        time_ranks = pd.Series(range(1, period + 1), index=x.index)
        corr = ranks.corr(time_ranks)
        return corr * 100 if not pd.isna(corr) else 0.0
    
    return series.rolling(period, min_periods=period).apply(rci_single, raw=False)

def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
    """计算布林带"""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return upper_band, rolling_mean, lower_band

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """计算MACD指标"""
    exp1 = series.ewm(span=fast).mean()
    exp2 = series.ewm(span=slow).mean()
    
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算ATR（真实波动范围）"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 新增：时间特征工程
    """
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 交易时段特征
        df['asian_session'] = ((df['hour'] >= 23) | (df['hour'] < 8)).astype(int)
        df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        
        # 周期性编码（避免边界问题）
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def calculate_price_position(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """计算价格在最近窗口内的相对位置"""
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    
    # 避免除零
    range_ = highest_high - lowest_low
    range_ = range_.replace(0, 1)  # 如果range为0，设为1避免除零
    
    return (close - lowest_low) / range_

def add_features_lstm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 确保有必要的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # 1. 技术指标
    print("   计算RSI...")
    df['RSI'] = calculate_rsi(df['close'])

    # MACD 指标
    print("   计算MACD...")
    macd, signal, hist = calculate_macd(df['close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist

    # 2. 波动率
    print("   计算波动率...")
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['Volatility'] = df['log_ret'].rolling(20).std() * np.sqrt(252 * 24 * 60)

    # 3. 移动平均线
    print("   计算移动平均线...")
    for period in [20, 50, 200]:
        df[f'MA{period}'] = df['close'].rolling(period).mean()
        df[f'MA{period}_Ratio'] = df['close'] / df[f'MA{period}']

    # 4. RCI
    print("   计算RCI...")
    df['RCI21'] = calculate_rci_vectorized(df['close'], 21)

    # 5. 布林带
    print("   计算布林带...")
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
    df['BB_upper'] = bb_upper
    df['BB_middle'] = bb_middle
    df['BB_lower'] = bb_lower
    df['BB_width'] = (bb_upper - bb_lower) / bb_middle

    # 6. 价格位置
    print("   计算价格位置...")
    df['Price_Position'] = calculate_price_position(df['high'], df['low'], df['close'])

    # 7. 成交量指标
    print("   计算成交量指标...")
    df['Volume_MA'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
    df['Volume_Change'] = df['volume'].pct_change()

    # 8. 价格动量
    print("   计算价格动量...")
    df['Momentum_5'] = df['close'].pct_change(5)
    df['Momentum_10'] = df['close'].pct_change(10)
    df['Momentum_20'] = df['close'].pct_change(20)

    # 9. 价格范围
    print("   计算价格范围...")
    df['Daily_Range'] = (df['high'] - df['low']) / df['close']
    df['Range_MA'] = df['Daily_Range'].rolling(20).mean()
    df['Range_Ratio'] = df['Daily_Range'] / df['Range_MA']

    # 10. ATR
    print("   计算ATR...")
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    df['ATR_Ratio'] = df['ATR'] / df['close']

    # 11. 时间特征（如果有时间戳）
    if 'timestamp' in df.columns:
        print("   添加时间特征...")
        df = add_time_features(df)

    # 12. 为环境准备的特殊特征
    print("   准备环境特征...")
    # 这里添加任何环境需要的特殊特征
    
    # 填充NaN值
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"   特征工程完成，共生成 {len(df.columns)} 个特征")
    return df

# 为环境类提供辅助函数
def get_env_observation_features(df: pd.DataFrame, t: int, position_size: float = 0.0, 
                               entry_price: float = 0.0, margin_level: float = float('inf')) -> np.ndarray:
    """
    为环境观察向量生成特征
    """
    if t >= len(df):
        t = len(df) - 1
    
    row = df.iloc[t]
    close = row['close']
    
    # 计算未实现盈亏比率（如果持有仓位）
    unrealized_pnl_ratio = 0.0
    if position_size != 0 and entry_price != 0:
        if position_size > 0:
            unrealized_pnl_ratio = (close - entry_price) / entry_price
        else:
            unrealized_pnl_ratio = (entry_price - close) / entry_price
    
    # 确保所有需要的特征都存在
    features = [
        row.get('log_ret', 0.0),
        row.get('RSI', 50.0) / 100.0 - 0.5,  # 归一化到[-0.5, 0.5]
        row.get('MACD_Hist', 0.0),
        row.get('BB_width', 0.0),
        row.get('Volatility', 0.0),
        row.get('Volume_Ratio', 1.0) - 1.0,  # 归一化
        row.get('Price_Position', 0.5) - 0.5,  # 归一化到[-0.5, 0.5]
        (close - row.get('MA50', close)) / close if 'MA50' in row else 0.0,
        position_size / 0.5,  # 假设最大仓位为0.5
        unrealized_pnl_ratio,
        min(margin_level / 100.0, 5.0) if margin_level != float('inf') else 5.0,  # 截断到5.0
        row.get('RCI21', 0.0) / 100.0  # 归一化到[-1, 1]
    ]
    
    return np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=5.0, neginf=-5.0)