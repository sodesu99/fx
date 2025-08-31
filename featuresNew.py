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

def add_features(df0: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 改进版：特征工程主函数
    """
    df = df0.copy()
    
    # 确保数据类型正确
    numeric_cols = ['close', 'high', 'low', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("🔧 开始特征工程...")
    
    # ✅ 1. 优化版RCI指标
    print("   计算RCI指标...")
    # df['RCI9'] = calculate_rci_vectorized(df['close'], 9)
    # df['RCI14'] = calculate_rci_vectorized(df['close'], 14)
    df['RCI21'] = calculate_rci_vectorized(df['close'], 21)
    
    # # ✅ 2. 新增：布林带
    # print("   计算布林带...")
    # bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
    # df['BB_upper'] = bb_upper
    # df['BB_middle'] = bb_middle
    # df['BB_lower'] = bb_lower
    # df['BB_width'] = (bb_upper - bb_lower) / bb_middle
    # df['BB_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # # ✅ 3. 新增：MACD
    # print("   计算MACD...")
    # macd, signal, hist = calculate_macd(df['close'])
    # df['MACD'] = macd
    # df['MACD_signal'] = signal
    # df['MACD_hist'] = hist
    # df['MACD_crossover'] = ((macd > signal) & (macd.shift() <= signal.shift())).astype(int)
    
    # # ✅ 4. 新增：ATR
    # print("   计算ATR...")
    # if all(col in df.columns for col in ['high', 'low']):
    #     df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    #     df['ATR_ratio'] = df['ATR'] / df['close']  # 相对ATR
    
    # # ✅ 5. 新增：RSI
    # print("   计算RSI...")
    # df['RSI'] = calculate_rsi(df['close'])
    # df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
    # df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
    
    # # 6. 原有指标（改进）
    # print("   计算基础指标...")
    # df['Volatility'] = df['close'].pct_change().rolling(20).std()
    
    
    # # 多周期移动平均
    # for period in [20, 50, 200]:
    #     df[f'MA{period}'] = df['close'].rolling(period).mean()
    #     df[f'MA{period}_Ratio'] = df['close'] / df[f'MA{period}']

    df['RCI9'] = 0
    df['RCI14'] = 0
    df['MA50_Ratio'] =0
    df['Volatility'] = 0

    # # ✅ 7. 改进：价格动量指标
    # print("   计算动量指标...")
    # for period in [5, 10, 20]:
    #     df[f'Price_Change_{period}'] = df['close'].pct_change(period)
    #     df[f'Price_Momentum_{period}'] = (df['close'] / df['close'].shift(period) - 1)
    
    # # ✅ 8. 新增：成交量指标
    # if 'volume' in df.columns:
    #     print("   计算成交量指标...")
    #     df['Volume_MA'] = df['volume'].rolling(20).mean()
    #     df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
    #     df['Price_Volume'] = df['close'].pct_change() * df['volume']  # 价量配合
    
    # ✅ 9. 新增：时间特征
    print("   添加时间特征...")
    df = add_time_features(df)
    
    # 10. 贝叶斯先验（保留原有逻辑）
    df['Bayesian_Prob'] = 0.5
    df['Bayesian_Prob'] = df['Bayesian_Prob'].fillna(0.5)
    
    # ✅ 11. 改进：动态归一化窗口
    window = min(100, len(df) // 10)  # 动态窗口大小
    
    print(f"   使用窗口大小: {window}")
    
    # 高低价范围
    df['High_Low_Range'] = (df['high'] - df['low']) / df['close']
    
    # 归一化处理（使用滚动窗口）
    df['close_norm'] = (df['close'] - df['close'].rolling(window).min()) / \
                       (df['close'].rolling(window).max() - df['close'].rolling(window).min() + 1e-8)
    
    if 'volume' in df.columns:
        df['volume_norm'] = (df['volume'] - df['volume'].rolling(window).min()) / \
                           (df['volume'].rolling(window).max() - df['volume'].rolling(window).min() + 1e-8)
    
    # ✅ 12. 特征选择和清理
    print("   清理特征...")
    
    # 移除中间计算列
    cols_to_drop = ['BB_upper', 'BB_lower', 'Volume_MA'] + [f'MA{p}' for p in [20, 200]]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 处理无穷值和NaN
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    
    # 记录清理前的长度
    initial_len = len(df)
    df = df.dropna()
    final_len = len(df)
    
    print(f"✅ 特征工程完成！")
    print(f"   - 数据长度: {initial_len} → {final_len} (丢弃 {initial_len - final_len} 行)")
    print(f"   - 特征维度: {len(df.columns)}")
    
    # ✅ 13. 特征重要性提示
    important_features = [
        'RCI9', 'RCI14', 'RCI21', 'BB_position', 'BB_width', 
        'MACD', 'MACD_hist', 'RSI', 'ATR_ratio', 'Volatility',
        'MA50_Ratio', 'close_norm', 'volume_norm', 'High_Low_Range',
        'Bayesian_Prob'
    ]
    
    available_features = [f for f in important_features if f in df.columns]
    print(f"   - 核心特征: {available_features}")
    
    return df

def get_feature_names() -> list:
    """
    ✅ 新增：获取特征名称列表（用于模型输入）
    """
    return [
        'RCI9', 'RCI14', 'RCI21', 'BB_position', 'BB_width',
        'MACD', 'MACD_hist', 'RSI', 'ATR_ratio', 'Volatility',
        'MA50_Ratio', 'close_norm', 'volume_norm', 'High_Low_Range',
        'Bayesian_Prob'
    ]