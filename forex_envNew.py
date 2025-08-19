# forex_env.py
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

class ForexEnv(gym.Env):
    """
    外汇交易环境（支持不确定性感知、动态仓位、风险调整奖励）
    动作空间: 0=hold, 1=buy, 2=sell
    状态空间: 技术指标 + 仓位状态
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        max_position: float = 0.3,        # 最大仓位比例（如 0.3 = 30% 资金）
        cost_ratio: float = 0.0001,       # 交易成本（点差+手续费）
        lookback: int = 50,               # 最小观察窗口
        uncertainty_scale: float = 0.5,   # 不确定性归一化系数：std > scale → confidence=0
        risk_weight: float = 0.1,         # Calmar 比率在奖励中的权重
        trade_penalty: float = 0.001,     # 每次交易惩罚
        window_size: int = 50,            # 计算风险指标的滑动窗口
    ):
        super(ForexEnv, self).__init__()

        # 参数校验
        assert max_position > 0, "max_position 必须大于 0"
        assert 0 <= cost_ratio < 0.01, "cost_ratio 应在 [0, 0.01) 之间"
        assert len(df) > lookback, "数据长度必须大于 lookback"

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.cost_ratio = cost_ratio
        self.lookback = lookback
        self.uncertainty_scale = uncertainty_scale
        self.risk_weight = risk_weight
        self.trade_penalty = trade_penalty
        self.window_size = window_size

        # 动作空间
        self.action_space = gym.spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell

        # 状态空间（11维）
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(11,), dtype=np.float32
        )

        # 环境状态变量（在 reset 中初始化）
        self.t = 0
        self.done = False
        self.balance = 0.0
        self.equity = 0.0
        self.net_worth = 0.0
        self.position_type = 0      # 0=none, 1=long, -1=short
        self.position_size = 0.0    # 仓位比例 [0, max_position]
        self.entry_price = 0.0
        self.returns_history = []

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> tuple[np.ndarray, dict]:
        """
        重置环境
        """
        super().reset(seed=seed)
        self.t = self.lookback
        self.done = False
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.net_worth = self.initial_balance
        self.position_type = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.returns_history = []  # 清空历史收益

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """
        获取当前状态向量
        """
        row = self.df.iloc[self.t]
        return np.array([
            row['RCI9'] / 100,              # RCI9 [-1,1]
            row['RCI14'] / 100,             # RCI14 [-1,1]
            row['RCI21'] / 100,             # RCI21 [-1,1]
            row['Bayesian_Prob'],           # 贝叶斯预测概率 [0,1]
            row['Volatility'],              # 波动率
            row['MA50_Ratio'] - 1,          # 价格偏离 MA50 比例
            row['close_norm'],              # 归一化收盘价
            row['volume_norm'],             # 归一化成交量
            row['High_Low_Range'],          # 高低波动范围
            float(self.position_type),      # 当前持仓方向
            self.position_size,             # 当前仓位大小
        ], dtype=np.float32)

    def step(
        self,
        action: int,
        uncertainty_std: float = 0.0
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步交易

        Args:
            action: 动作 (0=hold, 1=buy, 2=sell)
            uncertainty_std: 模型不确定性（来自 BNN），用于动态仓位

        Returns:
            obs, reward, terminated, truncated, info
        """
        # ✅ 提前检查：是否还能执行下一步？
        if self.t >= len(self.df) - 2:
            # 无法再执行有效交易
            obs = self._get_obs()
            reward = 0.0
            terminated = True
            truncated = True
            info = {
                'net_worth': self.net_worth,
                'equity': self.equity,
                'balance': self.balance,
                'position_type': self.position_type,
                'position_size': self.position_size,
                'uncertainty': uncertainty_std,
                'episode': {
                    'r': (self.net_worth / self.initial_balance - 1) * 100,
                    'l': self.t - self.lookback + 1,
                    't': self.t,
                    }
                }
            return obs, reward, terminated, truncated, info

        prev_equity = self.equity
        current_price = self.df.iloc[self.t + 1]['close']  # 下一时刻价格

        # --- 1. 计算综合置信度 ---
        # 将不确定性映射到 [0,1]：越小越可信
        confidence_from_uncertainty = 1.0 - min(uncertainty_std / self.uncertainty_scale, 1.0)
        bayes_prob = self.df.iloc[self.t]['Bayesian_Prob']
        overall_confidence = 0.7 * bayes_prob + 0.3 * confidence_from_uncertainty
        target_size = self.max_position * overall_confidence

        # --- 2. 执行交易逻辑 ---
        if action == 1 and self.position_type != 1:
            # 平空仓
            if self.position_type == -1:
                profit = self.position_size * self.equity * (self.entry_price - current_price) / self.entry_price
                self.balance += profit
            # 开多仓
            self.position_type = 1
            self.entry_price = current_price
            self.position_size = target_size
            self.balance *= (1 - self.cost_ratio)  # 扣除交易成本

        elif action == 2 and self.position_type != -1:
            # 平多仓
            if self.position_type == 1:
                profit = self.position_size * self.equity * (current_price - self.entry_price) / self.entry_price
                self.balance += profit
            # 开空仓
            self.position_type = -1
            self.entry_price = current_price
            self.position_size = target_size
            self.balance *= (1 - self.cost_ratio)  # 扣除交易成本

        # --- 3. 更新未实现盈亏和净值 ---
        if self.position_type == 1:
            unrealized_pnl = self.position_size * (current_price - self.entry_price) / self.entry_price
        elif self.position_type == -1:
            unrealized_pnl = self.position_size * (self.entry_price - current_price) / self.entry_price  # ✅ 修复：使用 entry_price
        else:
            unrealized_pnl = 0.0

        self.equity = self.balance * (1 + unrealized_pnl)
        self.net_worth = self.equity

        # --- 4. 计算基础收益 ---
        base_return = (self.equity - prev_equity) / (prev_equity + 1e-6)  # 防除零

        # --- 5. 奖励函数：正负分离 + 风险调整 ---
        if base_return > 0:
            reward = base_return * (1 + 2 * overall_confidence)  # 高置信正收益奖励更高
        else:
            reward = base_return * 0.5  # 缩小亏损惩罚

        # --- 6. 更新历史收益（用于风险指标）---
        self.returns_history.append(base_return)
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # --- 7. Calmar Ratio（年化收益 / 最大回撤）---
        calmar_ratio = 0.0
        if len(self.returns_history) > 1:
            cum_returns = np.cumprod([1 + r for r in self.returns_history])
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (peak - cum_returns) / (peak + 1e-6)
            max_drawdown = np.max(drawdown) if np.max(drawdown) > 0 else 1e-6
            annual_return = (cum_returns[-1] ** (252.0 * 60 / len(cum_returns))) - 1  # 分钟级数据
            calmar_ratio = annual_return / max_drawdown

        reward += self.risk_weight * calmar_ratio

        # --- 8. 交易频率惩罚 ---
        if action != 0:
            reward -= self.trade_penalty

        # --- 9. 下一步 ---
        self.t += 1

        # --- 10. 判断终止 ---
        truncated = False
        if self.t >= len(self.df) - 2:  # 防止 self.t+1 越界
            self.done = True
            truncated = True  # 数据结束

        terminated = self.done and not truncated

        # --- 11. 返回 ---
        obs = self._get_obs()
        info = {
            'net_worth': self.net_worth,
            'equity': self.equity,
            'balance': self.balance,
            'position_type': self.position_type,
            'position_size': self.position_size,
            'uncertainty': uncertainty_std,
            'calmar_ratio': calmar_ratio,
            'base_return': base_return,
            'overall_confidence': overall_confidence,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        渲染当前状态
        """
        if mode == 'human':
            row = self.df.iloc[self.t]
            print(
                f"T: {self.t:4d} | "
                f"Price: {row['close']:.5f} | "
                f"Net: {self.net_worth:8.1f} | "
                f"Pos: {self.position_type:+1d} | "
                f"Size: {self.position_size:4.2f} | "
                f"RCI9: {row['RCI9']:4.1f}"
            )