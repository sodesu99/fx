import numpy as np
import pandas as pd
import torch as th
import os
import gym
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from forex_envNew import ForexEnv
from bnn_policy import BNNActorCriticPolicy

# 设置随机种子
SEED = 42
set_random_seed(SEED, using_cuda=True)

# 1. 加载数据
df = pd.read_csv('./data/2024min1.csv', index_col=0, parse_dates=True)
df = df.iloc[-60000:]  # 取最近60000行
df.reset_index(inplace=True)

# 2. 生成特征
from featuresNew import add_features
df = add_features(df)
df.dropna(inplace=True)

# 3. 创建训练环境
env = ForexEnv(df, max_position=0.5)
env = Monitor(env)  # 添加监控

# 4. 测试环境
print("🧪 测试环境...")
test_env = ForexEnv(df.iloc[:1000], max_position=0.5)  # 使用小数据集测试
obs, _ = test_env.reset()
for _ in range(10):
    action = test_env.action_space.sample()
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        break
print("✅ 环境测试通过！")

# 5. 创建评估环境
df_eval = df.iloc[-10000:].copy()
eval_env = ForexEnv(df_eval, max_position=0.5)
eval_env = Monitor(eval_env)  # 添加监控

# 创建向量化评估环境
eval_vec_env = DummyVecEnv([lambda: eval_env])
eval_vec_env = VecNormalize(
    eval_vec_env,
    norm_obs=True,
    norm_reward=True,
    gamma=0.99,
    clip_obs=10.0,
    training=False  # 关键：设置为False表示不更新归一化参数
)

# 6. 回调函数
# 提前停止回调（当性能下降时停止）
stop_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=3,  # 连续3次评估无改进
    min_evals=5,                 # 至少5次评估后才检查
    verbose=1
)

eval_callback = EvalCallback(
    eval_vec_env,  # 使用向量化归一化环境
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    callback_after_eval=stop_callback,
    n_eval_episodes=1,
    verbose=1,
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints/",
    name_prefix="bnn_ppo_forex",
    verbose=1,
)

# 学习率调度器（带预热）
def linear_schedule(initial_lr, warmup_steps=5000):
    def schedule(progress_remaining):
        # 计算当前训练步数
        current_step = int((1 - progress_remaining) * total_timesteps)
        
        # 预热阶段
        if current_step < warmup_steps:
            return initial_lr * (current_step / warmup_steps)
        
        # 预热后线性衰减
        decay_progress = (current_step - warmup_steps) / (total_timesteps - warmup_steps)
        return initial_lr * (1 - decay_progress * 0.8)  # 衰减80%
    
    return schedule

# 设置总步数
total_timesteps = 200000

# 7. 构建模型
# 创建向量化训练环境
train_vec_env = DummyVecEnv([lambda: env])
train_vec_env = VecNormalize(
    train_vec_env,
    norm_obs=True,
    norm_reward=True,
    gamma=0.99,
    clip_obs=10.0
)

model = PPO(
    policy=BNNActorCriticPolicy,
    env=train_vec_env,
    verbose=1,
    tensorboard_log="./bnn_ppo_log/",
    learning_rate=linear_schedule(3e-4),
    batch_size=256,
    n_steps=1024,
    gamma=0.99,
    gae_lambda=0.9,
    ent_coef=0.01,
    clip_range=0.1,
    n_epochs=5,
    max_grad_norm=0.5,
    vf_coef=0.7,
    target_kl=0.03,
    device="cuda",
    seed=SEED,
    policy_kwargs={
        'log_std_init': -1.0,  # 更稳定的策略初始化
        'ortho_init': True,     # 正交初始化防止梯度爆炸
        'net_arch': [dict(pi=[256, 256], vf=[256, 256])]  # 明确网络架构
    }
)

print(f"💻 使用设备: {model.device}")
print(f"🔢 总训练步数: {total_timesteps}")
print("🚀 开始训练...")

# 8. 开始训练
model.learn(
    total_timesteps=total_timesteps,
    callback=[eval_callback, checkpoint_callback],
    tb_log_name=f"BNN_PPO_{os.getpid()}",
    progress_bar=True,
    reset_num_timesteps=False
)

# 9. 保存模型和归一化参数
model.save("bnn_ppo_forex_final")
train_vec_env.save("vec_normalize.pkl")  # 保存归一化参数
eval_vec_env.save("eval_vec_normalize.pkl")  # 保存评估环境归一化参数

print("✅ 训练完成！模型已保存。")
print(f"📊 查看TensorBoard: tensorboard --logdir=./bnn_ppo_log/")