
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

class ForexEnv(gym.Env):
    """
    ✅ 修复版：外汇交易环境
    主要修复：数据越界、盈亏计算、边界处理
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        max_position: float = 0.3,        # ✅ 降低默认最大仓位
        cost_ratio: float = 0.0001,
        lookback: int = 50,
        uncertainty_scale: float = 0.5,
        risk_weight: float = 0.1,
        trade_penalty: float = 0.01, # 交易频率惩罚
        window_size: int = 50,
    ):
        super(ForexEnv, self).__init__()

        # 参数校验
        assert max_position > 0, "max_position 必须大于 0"
        assert 0 <= cost_ratio < 0.01, "cost_ratio 应在 [0, 0.01) 之间"
        assert len(df) > lookback + 10, f"数据长度 {len(df)} 必须大于 {lookback + 10}"

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.cost_ratio = cost_ratio
        self.lookback = lookback
        self.uncertainty_scale = uncertainty_scale
        self.risk_weight = risk_weight
        self.trade_penalty = trade_penalty
        self.window_size = window_size

        # ✅ 计算有效交易范围
        self.max_t = len(self.df) - 2  # 确保不会越界

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
        """重置环境"""
        super().reset(seed=seed)
        self.t = self.lookback
        self.done = False
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.net_worth = self.initial_balance
        self.position_type = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.returns_history = []

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """
        ✅ 修复：添加边界检查
        """
        # 确保不越界
        t_safe = min(self.t, len(self.df) - 1)
        row = self.df.iloc[t_safe]
        
        obs = np.array([
            self.pnl_reward,
            row['RCI9'] / 100,
            row['Bayesian_Prob'],
            row['Volatility'],
            row['MA50_Ratio'] - 1,
            row['close_norm'],
            row['volume_norm'],
            row['High_Low_Range'],
            float(self.position_type),
            self.position_size,
        ], dtype=np.float32)
        
        # ✅ 处理 NaN 值
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _handle_termination(self):
        """处理环境终止情况"""
        self.done = True
        obs = self._get_obs()
        
        # 平仓处理
        final_reward = 0.0
        if self.position_type != 0:
            current_price = self.df.iloc[min(self.t, len(self.df) - 1)]['close']
            # 强制平仓
            if self.position_type == 1:
                profit = self.balance * self.position_size * (current_price - self.entry_price) / self.entry_price
            else:
                profit = self.balance * self.position_size * (self.entry_price - current_price) / self.entry_price
            
            self.balance += profit
            self.equity = self.balance
            self.net_worth = self.balance
            final_reward = profit / self.initial_balance
        
        info = {
            'net_worth': self.net_worth,
            'equity': self.equity,
            'balance': self.balance,
            'position_type': 0,  # 已平仓
            'position_size': 0.0,
            'uncertainty': 0.0,
            'episode': {
                'r': (self.net_worth / self.initial_balance - 1) * 100,
                'l': self.t - self.lookback + 1,
                't': self.t,
            }
        }
        
        return obs, final_reward, True, True, info

    def step(
        self,
        action: int,
        uncertainty_std: float = 0.0
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        ✅ 修复版：执行一步交易
        """
        # ✅ 边界检查
        if self.t >= self.max_t:
            return self._handle_termination()

        prev_equity = self.equity
        
        # ✅ 安全获取下一时刻价格
        try:
            current_price = self.df.iloc[self.t + 1]['close']
        except IndexError:
            return self._handle_termination()

        # 1. 计算综合置信度
        confidence_from_uncertainty = 1.0 - min(uncertainty_std / self.uncertainty_scale, 1.0)
        bayes_prob = self.df.iloc[min(self.t, len(self.df) - 1)]['Bayesian_Prob']
        overall_confidence = 0.7 * bayes_prob + 0.3 * confidence_from_uncertainty
        target_size = self.max_position * overall_confidence

        # 2. 执行交易逻辑
        trade_executed = False
        
        # 13. 返回
        obs = self._get_obs()
        info = {
            'net_worth': self.net_worth,
            'equity': self.equity,
            'balance': self.balance,
            'position_type': self.position_type,
            'position_size': self.position_size,
            'uncertainty': uncertainty_std,
            'overall_confidence': overall_confidence,
            'trade_executed': trade_executed,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """渲染当前状态"""
        if mode == 'human':
            t_safe = min(self.t, len(self.df) - 1)
            row = self.df.iloc[t_safe]
            print(
                f"T: {self.t:4d} | "
                f"Price: {row['close']:.5f} | "
                f"Net: {self.net_worth:8.1f} | "
                f"Pos: {self.position_type:+1d} | "
                f"Size: {self.position_size:4.2f} | "
                f"RCI9: {row['RCI9']:4.1f}"
            )