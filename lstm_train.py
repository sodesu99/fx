import numpy as np
import pandas as pd
import torch as th
import os
from typing import Callable, Dict, Any
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym

# Import custom modules
from featuresNew import add_features_lstm
from LSTM_bnn_policy import LSTM_BNN_Policy
from MarginForexEnv import MarginForexEnv

# -------------------------------
# 🔧 Step 0: 设置随机种子
# -------------------------------

SEED = 42
set_random_seed(SEED, using_cuda=True)
np.random.seed(SEED)
th.manual_seed(SEED)
if th.cuda.is_available():
    th.cuda.manual_seed_all(SEED)

print("🚀 开始训练 LSTM-BNN-SAC 外汇交易模型")
print("=" * 60)

# -------------------------------
# 📂 Step 1: 加载和预处理数据
# -------------------------------

print("📂 加载数据...")
df = pd.read_csv('./data/2024min1.csv', index_col=0, parse_dates=True)

# 限制数据量（避免内存问题）
data_size = min(10000, len(df))
df = df.iloc[-data_size:].copy()
df.reset_index(inplace=True)
df.rename(columns={'index': 'timestamp'}, inplace=True)  # 确保有时间列

print(f"   原始数据: {len(df):,} 行")

# -------------------------------
# 🔧 Step 2: 特征工程（确保包含必要字段）
# -------------------------------

print("🔧 特征工程...")


df = add_features_lstm(df)
print(f"   处理后: {len(df):,} 行, {len(df.columns)} 列")

# -------------------------------
# 🔀 Step 3: 时序分割
# -------------------------------

def create_data_splits(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"📊 数据分割:")
    print(f"   训练集: {len(df_train):,} 行 ({train_ratio*100:.0f}%)")
    print(f"   验证集: {len(df_val):,} 行 ({val_ratio*100:.0f}%)")
    print(f"   测试集: {len(df_test):,} 行 ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return df_train, df_val, df_test

df_train, df_val, df_test = create_data_splits(df)

# -------------------------------
# 🧪 Step 4: 环境安全性测试
# -------------------------------

def test_environment_safety(env_class, df_sample: pd.DataFrame):
    print("🧪 开始环境安全性测试...")
    
    test_env = env_class(df_sample, max_position_ratio=0.3, lookback=60)
    obs, _ = test_env.reset()
    
    issues = []
    
    for i in range(min(100, len(df_sample) - test_env.lookback - 10)):
        action = test_env.action_space.sample()
        
        try:
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                issues.append(f"步骤 {i}: 观测包含 NaN/Inf")
            if np.isnan(reward) or np.isinf(reward):
                issues.append(f"步骤 {i}: 奖励为 NaN/Inf: {reward}")
            if info['equity'] <= 0:
                issues.append(f"步骤 {i}: 权益为负: {info['equity']}")
                
            if terminated or truncated:
                break
                
        except Exception as e:
            issues.append(f"步骤 {i}: 异常 {str(e)}")
            break
    
    if issues:
        print("⚠️ 发现环境问题:")
        for issue in issues[:5]:
            print(f"   - {issue}")
        return False
    else:
        print("✅ 环境安全性测试通过!")
        return True

# MarginForexEnv is now imported at the top of the file

if not test_environment_safety(MarginForexEnv, df_train.iloc[:1000]):
    raise RuntimeError("环境测试失败，请检查 MarginForexEnv 实现")

# -------------------------------
# 🏗️ Step 5: 创建训练/验证环境
# -------------------------------

print("🏗️ 创建训练环境...")

train_env = MarginForexEnv(
    df=df_train,
    initial_balance=100_000,
    leverage=100,
    max_position_ratio=0.5,
    cost_ratio=7e-5,
    lookback=60,
    stop_out_level=50.0,
)
train_env = Monitor(train_env, filename="logs/train_monitor")

eval_env = MarginForexEnv(
    df=df_val,
    initial_balance=100_000,
    leverage=100,
    max_position_ratio=0.5,
    cost_ratio=7e-5,
    lookback=60,
    stop_out_level=50.0,
)
eval_env = Monitor(eval_env, filename="logs/eval_monitor")

# -------------------------------
# 🔄 Step 6: 向量化 + 归一化
# -------------------------------

# LSTM 要求序列输入，使用 DummyVecEnv 包装
train_vec_env = DummyVecEnv([lambda: train_env])

# ✅ 重要：VecNormalize 不会打乱序列维度
# 我们只归一化观测中的特征，不归一化整个序列结构
train_vec_env = VecNormalize(
    train_vec_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99,
    epsilon=1e-8,
)

eval_vec_env = DummyVecEnv([lambda: eval_env])
eval_vec_env = VecNormalize(
    eval_vec_env,
    norm_obs=True,
    norm_reward=False,
    training=False,
    clip_obs=10.0,
    epsilon=1e-8,
)

# -------------------------------
# 🧠 Step 7: 自定义 LSTM-BNN 组件
# -------------------------------


# -------------------------------
# ⚙️ Step 8: 创建 SAC 模型
# -------------------------------

print("🤖 创建 SAC 模型（适配 LSTM-BNN）...")

model = SAC(
    policy=LSTM_BNN_Policy,
    env=train_vec_env,
    verbose=1,
    tensorboard_log="./sac_lstm_bnn_log/",
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=100_000,
    learning_starts=5000,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    device="cuda" if th.cuda.is_available() else "cpu",
    seed=SEED,
    policy_kwargs={
        'log_std_init': -2.0,
        'net_arch': dict(pi=[256, 256], qf=[256, 256]),
        'share_features_extractor': False,
    }
)

print(f"💻 使用设备: {model.device}")
total_params = sum(p.numel() for p in model.policy.parameters())
print(f"📊 模型参数: {total_params:,}")

# -------------------------------
# 📈 Step 9: 回调函数
# -------------------------------

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

eval_callback = EvalCallback(
    eval_vec_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10_000,
    deterministic=True,
    n_eval_episodes=3,
    verbose=1,
)

checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path="./checkpoints/",
    name_prefix="lstm_bnn_sac",
    verbose=1,
)

# -------------------------------
# 🚀 Step 10: 开始训练
# -------------------------------

total_timesteps = 50_000
print(f"\n🏋️ 开始训练 {total_timesteps:,} 步...")

try:
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=f"LSTM_BNN_SAC_{SEED}",
        progress_bar=True,
    )
    print("✅ 训练成功完成！")

except KeyboardInterrupt:
    print("\n⚠️ 训练被中断，保存当前模型...")
except Exception as e:
    print(f"\n❌ 训练出错: {e}")
    import traceback
    traceback.print_exc()

# -------------------------------
# 💾 Step 11: 保存模型
# -------------------------------

print("💾 保存最终模型...")
model.save("lstm_bnn_sac_forex_final")
train_vec_env.save("train_vec_normalize.pkl")
eval_vec_env.save("eval_vec_normalize.pkl")

df_test.to_csv("data/test_data.csv", index=False)
print("✅ 所有文件保存完成！")

# -------------------------------
# 📊 提示信息
# -------------------------------

print("\n" + "=" * 60)
print("🎉 训练流程完成！")
print("=" * 60)
print("📊 查看日志:")
print("   tensorboard --logdir=./sac_lstm_bnn_log/")
print("🚀 评估模型:")
print("   python evaluate.py")
print("=" * 60)