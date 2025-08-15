# forex_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ForexEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, leverage=30, cost_per_trade=0.00007):
        super(ForexEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.cost_per_trade = cost_per_trade  # 7 pip spread approx
        self.n_steps = len(df)
        self.lookback = 50  # 用过去50小时数据

        # 动作空间：0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # 状态空间：RCI, Bayesian_Prob, Volatility, etc.
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(10,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback
        self.done = False
        self.balance = self.initial_balance
        self.holdings = 0.0
        self.position = 0  # -1=sell, 0=hold, +1=buy
        self.net_worth = self.initial_balance
        self.max_drawdown = 0
        self.prev_net_worth = self.initial_balance
        return self._get_state(), {}

    def _get_state(self):
        # 提取特征
        row = self.df.iloc[self.t]
        features = np.array([
            row['RCI9'] / 100,
            row['RCI14'] / 100,
            row['RCI21'] / 100,
            row['Bayesian_Prob'],
            row['Volatility'],
            row['MA50_Close_Ratio'] - 1,
            self.position,
            np.log(self.net_worth / self.initial_balance),
            (self.df['close'].iloc[self.t] - self.df['close'].iloc[self.t-1]) / self.df['close'].iloc[self.t-1],
            row['volume'] / 1e6
        ], dtype=np.float32)
        return features

    def step(self, action):
        prev_net_worth = self.net_worth
        reward = 0.0

        # 执行交易
        current_price = self.df.iloc[self.t]['close']

        if action == 1 and self.position != 1:  # 买入
            if self.position == -1:  # 平空
                reward += (self.holdings * (current_price - self.entry_price)) * self.leverage
                self.balance += reward
            self.entry_price = current_price
            self.holdings = self.balance / current_price
            self.position = 1
            self.balance -= self.cost_per_trade * self.balance  # 手续费

        elif action == 2 and self.position != -1:  # 卖出
            if self.position == 1:  # 平多
                reward += (self.holdings * (current_price - self.entry_price)) * self.leverage
                self.balance += reward
            self.entry_price = current_price
            self.holdings = self.balance / current_price
            self.position = -1
            self.balance -= self.cost_per_trade * self.balance

        # 更新净值
        if self.position == 1:
            self.net_worth = self.balance + self.holdings * (current_price - self.entry_price) * self.leverage
        elif self.position == -1:
            self.net_worth = self.balance + self.holdings * (self.entry_price - current_price) * self.leverage
        else:
            self.net_worth = self.balance

        # 奖励函数（可加入贝叶斯权重）
        reward = (self.net_worth - prev_net_worth) / prev_net_worth
        reward -= 0.0001  # 小惩罚，避免无意义交易

        # 贝叶斯置信加权奖励（可选）
        bayes_prob = self.df.iloc[self.t]['Bayesian_Prob']
        reward = reward * (0.5 + 1.5 * bayes_prob)  # 高置信度信号奖励更高

        # 更新时间
        self.t += 1
        if self.t >= self.n_steps - 1:
            self.done = True

        return self._get_state(), reward, self.done, False, {}

    def render(self, mode='human'):
        print(f"Step: {self.t}, Net Worth: {self.net_worth:.2f}, Position: {self.position}")

# 使用示例
# env = ForexEnv(df)
# obs, _ = env.reset()
# for _ in range(100):
#     obs, reward, done, _, _ = env.step(env.action_space.sample())
#     if done:
#         break