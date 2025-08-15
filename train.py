# import pandas as pd
# from stable_baselines3 import PPO

# from features import generate_features
# from forex_env import ForexEnv

# df = pd.read_csv('./data/data2024.csv', index_col='timestamp', parse_dates=True)
# df = generate_features(df)
# print(df[['close', 'RCI9', 'RCI14', 'RCI21', 'Bayesian_Prob']].tail())



# train.py
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from forex_envNew import ForexEnv
from bnn_policy import BNNActorCriticPolicy

# 1. 加载数据
df = pd.read_csv('./data/2024min1.csv', index_col=0, parse_dates=True)
df = df.iloc[-60000:]  # 取最近60000行数据
df.reset_index(inplace=True)

# 2. 生成特征
from featuresNew import add_features
df = add_features(df)

# 确保数据完整
df.dropna(inplace=True)

# 3. 创建环境
env = ForexEnv(df, max_position=0.5)

# 4. 训练模型（使用标准PPO，但策略由BNN控制）
# 注意：Stable-Baselines3 不直接支持 MC Dropout 推理
# 我们先训练一个标准模型，后续用 BNN 推理时再注入不确定性

model = PPO(
    BNNActorCriticPolicy,
    env,
    verbose=1,
    tensorboard_log="./bnn_ppo_log/",
    learning_rate=5e-4,          # 学习率调高，加快策略变化
    batch_size=64,
    n_steps=256,
    gamma=0.95,                  # 折扣因子降低，重视短期回报
    ent_coef=0.05,               # 加熵奖励，鼓励策略多样性
    clip_range=0.5,               # 允许策略更新幅度更大
    device="cuda"  # 启用GPU
)

print("🚀 开始训练...")
model.learn(total_timesteps=200000)
model.save("bnn_ppo_forex")

print("✅ 训练完成！模型已保存。")

# 5. 测试（演示不确定性使用）
# train.py
obs, _ = env.reset()
for _ in range(1000):
    # 获取动作（由 PPO 决定）
    action, _ = model.predict(obs, deterministic=False)

    # 获取不确定性（用于动态仓位）
    feature_mean, feature_std = model.policy.predict_with_uncertainty(obs, n_samples=10)
    uncertainty_std = feature_std.mean().item()  # 取平均不确定性

    # 传入环境进行动态仓位管理
    obs, reward, done, _, info = env.step(action, uncertainty_std=uncertainty_std)

    if done:
        break
env.render()