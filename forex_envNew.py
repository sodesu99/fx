# forex_env.py
import gymnasium as gym
import numpy as np
import pandas as pd

class ForexEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, max_position=0.3, cost_ratio=0.0001):
        super(ForexEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_position = max_position  # 最大仓位（如 0.1 = 10% 资金）
        self.cost_ratio = cost_ratio
        self.n_steps = len(df)
        self.lookback = 50

        # 动作空间：0=hold, 1=buy, 2=sell
        self.action_space = gym.spaces.Discrete(3)

        # 状态空间
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(11,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback
        self.done = False
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position_size = 0.0  # 当前仓位比例
        self.position_type = 0    # 0=none, 1=long, -1=short
        self.entry_price = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.t]
        return np.array([
            row['RCI9'] / 100,
            row['RCI14'] / 100,
            row['RCI21'] / 100,
            row['Bayesian_Prob'],
            row['Volatility'],
            row['MA50_Ratio'] - 1,
            row['close_norm'] ,  
            row['volume_norm'] ,
            row['High_Low_Range'] ,
            self.position_type,
            self.position_size,
        ], dtype=np.float32)

    def step(self, action, uncertainty_std=0.0):
        prev_equity = self.equity
        current_price = self.df.iloc[self.t + 1]['close']

        # === 动态仓位管理模块 ===
        confidence = 1.0 - min(uncertainty_std / 0.5, 1.0)  # 0~1
        bayes_prob = self.df.iloc[self.t]['Bayesian_Prob']
        overall_confidence = 0.7 * bayes_prob + 0.3 * confidence
        target_size = self.max_position * overall_confidence

        # === 执行交易逻辑 ===
        if action == 1 and self.position_type != 1:
            # 平空
            if self.position_type == -1:
                profit = self.position_size * self.equity * (self.entry_price - current_price) / self.entry_price
                self.balance += profit
            # 开多
            self.position_type = 1
            self.entry_price = current_price
            self.position_size = target_size
            self.balance *= (1 - self.cost_ratio)

        elif action == 2 and self.position_type != -1:
            # 平多
            if self.position_type == 1:
                profit = self.position_size * self.equity * (current_price - self.entry_price) / self.entry_price
                self.balance += profit
            # 开空
            self.position_type = -1
            self.entry_price = current_price
            self.position_size = target_size
            self.balance *= (1 - self.cost_ratio)

        # === 更新净值 ===
        if self.position_type == 1:
            unrealized = self.position_size * (current_price - self.entry_price) / self.entry_price
        elif self.position_type == -1:
            unrealized = self.position_size * (self.entry_price - current_price) / current_price
        else:
            unrealized = 0

        self.equity = self.balance * (1 + unrealized)
        self.net_worth = self.equity

        # === 计算基础收益 ===
        base_reward = (self.equity - prev_equity) / prev_equity

        # === 正负收益分开处理 ===
        if base_reward > 0:
            reward = base_reward * (1 + 2 * overall_confidence)
        else:
            reward = base_reward * 0.5  # 缩小负收益惩罚

        # === 记录历史收益用于风险指标 ===
        if not hasattr(self, "returns_history"):
            self.returns_history = []
        self.returns_history.append(base_reward)
        window_size = 50
        if len(self.returns_history) > window_size:
            self.returns_history.pop(0)

        # === 计算 Calmar 比率 ===
        if len(self.returns_history) > 1:
            cum_returns = np.cumprod([1 + r for r in self.returns_history])
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (peak - cum_returns) / peak
            max_drawdown = np.max(drawdown) if np.max(drawdown) > 0 else 1e-6
            annual_return = (cum_returns[-1] ** (252 / len(cum_returns))) - 1
            calmar_ratio = annual_return / max_drawdown
        else:
            calmar_ratio = 0

        # === 将风险指标纳入奖励 ===
        reward += 0.1 * calmar_ratio  # 0.1 为风险权重，可调节

        # === 频繁交易惩罚 ===
        if action != 0:
            reward -= 0.001

        # === 下一步 ===
        self.t += 1
        if self.t >= self.n_steps - 1:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}


    def render(self, mode='human'):
        print(f"T: {self.t} | Net: {self.net_worth:.0f} | Pos: {self.position_type:+.0f} | Size: {self.position_size:.2f}")