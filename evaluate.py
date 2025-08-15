# evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from forex_envNew import ForexEnv
from bnn_policy import BNNActorCriticPolicy
import os

# ----------------------------
# 1. 加载评估数据
# ----------------------------
df_eval = pd.read_csv('data/2024min1.csv', index_col=0, parse_dates=True)
df_eval = df_eval.iloc[5000:6000]  # 取500数据
df_eval.reset_index(inplace=True)

# 添加特征（使用之前的 features.py）
from featuresNew import add_features
df_eval = add_features(df_eval)

# ----------------------------
# 2. 创建评估环境
# ----------------------------
env = ForexEnv(df_eval, initial_balance=10000)

# ----------------------------
# 3. 加载训练好的模型
# ----------------------------
if not os.path.exists("bnn_ppo_forex.zip"):
    raise FileNotFoundError("请先运行 train.py 生成模型")

model = PPO.load("bnn_ppo_forex", env=env, custom_objects={"policy_class": BNNActorCriticPolicy})

print("✅ 模型加载成功！开始评估...")

# ----------------------------
# 4. 运行回测并记录轨迹
# ----------------------------
obs, _ = env.reset()
done = False
rewards = []
net_worths = []
positions = []
position_sizes = []
actions = []
uncertainties = []

while not done:
    # 获取动作
    action, _ = model.predict(obs, deterministic=True)

    # 获取不确定性（用于分析）
    feature_mean, feature_std = model.policy.predict_with_uncertainty(obs, n_samples=10)
    uncertainty = feature_std.mean().item()

    # 执行 step
    obs, reward, done, _, info = env.step(action, uncertainty_std=uncertainty)

    # 记录
    rewards.append(reward)
    net_worths.append(env.net_worth)
    positions.append(env.position_type)
    position_sizes.append(env.position_size)
    actions.append(action)
    uncertainties.append(uncertainty)

    # 打印价格与RCI（以RCI9为例，可加RCI14/21等）
    if action == 1 or action == 2:
        t = env.t  # 当前环境步数
        row = env.df.iloc[t]
        print(f"{'Buy' if action == 1 else 'Sell'}信号: 时间={row['timestamp'] if 'timestamp' in row else t}, 价格={row['close']:.5f}, RCI9={row['RCI9']:.2f}, RCI14={row['RCI14']:.2f}, RCI21={row['RCI21']:.2f}")


net_worths = np.array(net_worths)
positions = np.array(positions)
position_sizes = np.array(position_sizes)
uncertainties = np.array(uncertainties)
rewards = np.array(rewards)

print(f"✅ 回测完成！共 {len(net_worths)} 个时间步")



# 在 evaluate.py 末尾添加此函数
def plot_trading_signals(df, env, actions, positions, net_worths, save_path="trading_signals.png"):
    """
    绘制交易信号叠加在价格图上
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax1 = plt.subplots(figsize=(16, 8))

     # 只绘制后500行数据
    # 1. 绘制价格曲线
    ax1.plot(df.index, df['close'], label='EUR/USD Price', color='blue', alpha=0.9, linewidth=1)

    # 2. 标出买入（绿色三角）和卖出（红色倒三角）
    buy_signals = [(i, df['close'].iloc[i + 50]) for i, a in enumerate(actions) if a == 1]
    sell_signals = [(i, df['close'].iloc[i + 50]) for i, a in enumerate(actions) if a == 2]

    if buy_signals:
        idx, prices = zip(*buy_signals)
        ax1.scatter(idx, prices, marker='^', color='green', s=100, label='Buy Signal', zorder=5)

    if sell_signals:
        idx, prices = zip(*sell_signals)
        ax1.scatter(idx, prices, marker='v', color='red', s=100, label='Sell Signal', zorder=5)

    # 3. 添加持有状态背景色
    ax1.fill_between(
        range(len(positions)), df['close'].min(), df['close'].max(),
        where=np.array(positions) == 1,
        color='green', alpha=0.1, label='Long Position'
    )
    ax1.fill_between(
        range(len(positions)), df['close'].min(), df['close'].max(),
        where=np.array(positions) == -1,
        color='red', alpha=0.1, label='Short Position'
    )

    # 4. 设置标签和标题
    ax1.set_title('EUR/USD Trading Signals with PPO + BNN Strategy', fontsize=16)
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 5. 添加第二轴：净值曲线
    ax2 = ax1.twinx()
    ax2.plot(net_worths, label='Net Worth', color='purple', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Net Worth', fontsize=12)
    ax2.legend(loc='upper right')

    # 6. 优化 x 轴标签（如果 index 是时间）
    try:
        ax1.set_xticks(np.arange(0, len(df), max(1, len(df)//10)))
        ax1.set_xticklabels([df.index[i] for i in np.arange(0, len(df), max(1, len(df)//10))], rotation=45)
    except:
        pass  # 如果索引不是时间，跳过

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"✅ 交易信号图已保存: {save_path}")




# ----------------------------
# 5. 计算核心绩效指标
# ----------------------------
initial_balance = 10000
final_balance = net_worths[-1]
total_return = (final_balance - initial_balance) / initial_balance

winning_trades = [r for r in rewards if r > 0]
losing_trades = [r for r in rewards if r < 0]

win_rate = len(winning_trades) / len(rewards) if len(rewards) > 0 else 0
avg_win = np.mean(winning_trades) if winning_trades else 0
avg_loss = -np.mean(losing_trades) if losing_trades else 0
profit_factor = avg_win / avg_loss if avg_loss > 0 else np.inf

sharpe_ratio = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(25)  # 年化（25小时/天）

max_drawdown = np.max(np.maximum.accumulate(net_worths) - net_worths)
max_drawdown_pct = max_drawdown / np.max(np.maximum.accumulate(net_worths)) * 100

print("\n" + "="*40)
print("📊 模型性能报告")
print("="*40)
print(f"总回报:        {total_return*100:.2f}%")
print(f"年化夏普比率:   {sharpe_ratio:.2f}")
print(f"胜率:          {win_rate*100:.1f}%")
print(f"平均盈利:       {avg_win:.6f}")
print(f"平均亏损:       {avg_loss:.6f}")
print(f"盈亏比:         {profit_factor:.2f}")
print(f"最大回撤:       {max_drawdown:.0f} ({max_drawdown_pct:.1f}%)")
print(f"最终净值:       {final_balance:.0f}")
print("="*40)

print("动作分布 (0=hold, 1=buy, 2=sell):", np.bincount(actions, minlength=3))
print("平均仓位大小:", np.mean(position_sizes))
print("不确定性均值:", np.mean(uncertainties))


# ----------------------------
# 6. 可视化
# ----------------------------
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# 净值曲线
axs[0].plot(net_worths, label='Net Worth', color='blue')
axs[0].axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
axs[0].set_title("Net Worth Over Time")
axs[0].legend()
axs[0].grid(True)

# 仓位与动作
axs[1].plot(positions, label='Position (1=Long, -1=Short)', color='green', drawstyle='steps')
axs[1].plot(position_sizes, label='Position Size', color='orange', alpha=0.7)
axs[1].set_title("Position & Size")
axs[1].legend()
axs[1].grid(True)

# 不确定性
axs[2].plot(uncertainties, label='Model Uncertainty', color='red', alpha=0.7)
axs[2].set_title("Model Uncertainty (Std)")
axs[2].legend()
axs[2].grid(True)

# 收益分布
axs[3].hist(rewards, bins=50, alpha=0.7, color='purple')
axs[3].set_title("Reward Distribution")
axs[3].axvline(x=np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.6f}')
axs[3].legend()
axs[3].grid(True)

plt.xlabel("Time Step")
plt.tight_layout()
plt.savefig("evaluation_report.png", dpi=150)
# ----------------------------
# 8. 绘制交易信号图
# ----------------------------
plot_trading_signals(
    df=df_eval,
    env=env,
    actions=actions,
    positions=positions,
    net_worths=net_worths
)

# ----------------------------
# 7. 分析：不确定性是否帮助了风控？
# ----------------------------
high_uncertainty = uncertainties > np.median(uncertainties)
low_uncertainty = ~high_uncertainty

if high_uncertainty.sum() > 0 and low_uncertainty.sum() > 0:
    avg_reward_high = rewards[high_uncertainty].mean()
    avg_reward_low = rewards[low_uncertainty].mean()

    print("\n🔍 不确定性有效性分析")
    print(f"高不确定性时段平均收益: {avg_reward_high:.6f}")
    print(f"低不确定性时段平均收益: {avg_reward_low:.6f}")
    if avg_reward_low > avg_reward_high:
        print("✅ 模型在低不确定性时表现更好 → 不确定性估计有效！")
    else:
        print("⚠️  高不确定性时收益更高 → 需检查信号质量或模型校准")
else:
    print("\n⚠️  不确定性数据不足，无法分析")

