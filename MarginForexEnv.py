import numpy as np
import pandas as pd
from collections import deque
import gymnasium as gym
from featuresNew import get_env_observation_features


class MarginForexEnv(gym.Env):
    """
    支持保证金和强平机制的高频外汇交易环境
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100_000.0,   # 初始资金（USD）
        leverage: int = 100,                  # 杠杆倍数（如 1:100）
        max_position_ratio: float = 0.5,      # 单笔最大使用资金比例
        cost_ratio: float = 7e-5,             # 点差成本（0.7 pip）
        lookback: int = 60,
        margin_call_level: float = 100.0,     # 百分比：100% 触发警告
        stop_out_level: float = 50.0,         # 50% 强平
        trade_penalty: float = 0.0001,
    ):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.max_position_ratio = max_position_ratio
        self.cost_ratio = cost_ratio
        self.lookback = lookback
        self.margin_call_level = margin_call_level
        self.stop_out_level = stop_out_level
        self.trade_penalty = trade_penalty
        self.max_t = len(self.df) - 2

        # 动作空间：连续 [-1,1] → 控制方向和仓位大小
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(lookback, 12), dtype=np.float32
        )

        self.history_buffer = deque(maxlen=lookback)

        # --- 资金状态 ---
        self.t = 0
        self.balance = 0.0           # 已实现余额
        self.equity = 0.0            # 权益 = balance + unrealized_pnl
        self.used_margin = 0.0       # 已用保证金
        self.free_margin = 0.0       # 可用保证金
        self.margin_level = 0.0      # Margin Level (%)

        self.position_size = 0.0     # 当前净仓位（+多 -空）
        self.entry_price = 0.0       # 加权开仓价
        self.trades = 0
        self.episode_pnls = []

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.used_margin = 0.0
        self.free_margin = self.initial_balance
        self.margin_level = float('inf')
        self.position_size = 0.0
        self.entry_price = 0.0
        self.trades = 0
        self.episode_pnls = []
        self.history_buffer.clear()

        # 初始化历史窗口
        for i in range(self.lookback):
            idx = self.t - self.lookback + i
            obs_vec = self._get_observation_vector(idx)
            self.history_buffer.append(obs_vec)

        return self._get_obs(), {}

    def _get_observation_vector(self, t: int) -> np.ndarray:
        # 使用辅助函数生成观察向量
        return get_env_observation_features(
            self.df, t, 
            self.position_size, 
            self.entry_price, 
            self.margin_level
        )

    def _get_obs(self):
        return np.array(list(self.history_buffer))

    def _update_margin(self):
        """更新保证金相关指标"""
        current_price = self.df.iloc[min(self.t, len(self.df) - 1)]['close']

        # 计算未实现盈亏
        unrealized_pnl = 0.0
        if self.position_size != 0 and self.entry_price != 0:
            if self.position_size > 0:
                unrealized_pnl = self.position_size * self.initial_balance * \
                    (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl = self.position_size * self.initial_balance * \
                    (self.entry_price - current_price) / self.entry_price

        self.equity = self.balance + unrealized_pnl

        # 计算保证金（假设合约大小为 1 手 = $10万，但按比例缩放）
        # 保证金 = 名义价值 / 杠杆
        notional_value = abs(self.position_size) * self.initial_balance
        self.used_margin = notional_value / self.leverage
        self.free_margin = self.equity - self.used_margin

        # Margin Level = Equity / Used Margin × 100%
        if self.used_margin > 0:
            self.margin_level = (self.equity / self.used_margin) * 100.0
        else:
            self.margin_level = float('inf')  # 无仓位时无限高

    def _liquidate_all(self):
        """强制平仓所有持仓"""
        if self.position_size == 0:
            return

        current_price = self.df.iloc[min(self.t, len(self.df) - 1)]['close']
        if self.position_size > 0:
            realized_pnl = self.position_size * self.initial_balance * \
                (current_price - self.entry_price) / self.entry_price
        else:
            realized_pnl = self.position_size * self.initial_balance * \
                (self.entry_price - current_price) / self.entry_price

        self.balance += realized_pnl
        self.position_size = 0.0
        self.entry_price = 0.0

    def step(self, action: np.ndarray, uncertainty_std: float = 0.0):
        action = np.clip(action.item(), -1.0, 1.0)

        if self.t >= self.max_t:
            return self._handle_termination()

        try:
            current_price = self.df.iloc[self.t + 1]['close']
        except IndexError:
            return self._handle_termination()

        prev_equity = self.equity
        reward = 0.0
        trade_executed = False

        # --- 更新保证金状态 ---
        self._update_margin()

        # --- 检查是否触发强平 ---
        if self.used_margin > 0 and self.margin_level <= self.stop_out_level:
            self._liquidate_all()
            reward -= 1.0  # 严重惩罚
            self.t += 1
            obs = self._get_obs()
            info = {
                "net_worth": self.balance,
                "equity": self.equity,
                "balance": self.balance,
                "position_size": 0.0,
                "margin_call": True,
                "liquidation": True,
                "episode": {"r": (self.balance / self.initial_balance - 1) * 100}
            }
            return obs, reward, True, False, info

        # --- 正常交易逻辑 ---
        confidence_factor = 1.0 - min(uncertainty_std / 2.0, 0.9)
        target_position = action * self.max_position_ratio * confidence_factor

        delta = target_position - self.position_size
        threshold = 0.01 * self.max_position_ratio

        if abs(delta) > threshold:
            # 检查是否有足够保证金开仓
            required_margin = (abs(target_position) - abs(self.position_size)) * \
                            self.initial_balance / self.leverage
            if required_margin <= self.free_margin * 0.8:  # 保留缓冲
                cost = abs(delta) * self.initial_balance * self.cost_ratio
                if cost < self.balance * 0.01:
                    self.balance -= cost
                    self.position_size = target_position
                    self.entry_price = current_price
                    trade_executed = True
                    self.trades += 1
                    reward -= self.trade_penalty

        # 更新 margin（含未实现盈亏）
        self._update_margin()

        # 奖励函数（略）
        step_pnl = self.equity - prev_equity
        self.episode_pnls.append(step_pnl)
        reward += step_pnl / self.initial_balance * 0.1
        reward -= uncertainty_std * 0.01

        # 更新历史
        obs_vec = self._get_observation_vector(self.t)
        self.history_buffer.append(obs_vec)
        self.t += 1

        terminated = self.t >= self.max_t
        truncated = False

        info = {
            "net_worth": self.balance,
            "equity": self.equity,
            "balance": self.balance,
            "position_size": self.position_size,
            "margin_level": self.margin_level,
            "used_margin": self.used_margin,
            "free_margin": self.free_margin,
            "liquidation": False,
            "trade_executed": trade_executed,
            "trades": self.trades,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _handle_termination(self):
        self._update_margin()
        if self.used_margin > 0 and self.margin_level <= self.stop_out_level:
            self._liquidate_all()
        elif self.position_size != 0:
            current_price = self.df.iloc[min(self.t, len(self.df) - 1)]['close']
            if self.position_size > 0:
                pnl = (current_price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - current_price) / self.entry_price
            self.balance += self.position_size * self.initial_balance * pnl
            self.position_size = 0.0

        return self._get_obs(), 0.0, True, True, {
            "net_worth": self.balance,
            "equity": self.balance,
            "balance": self.balance,
            "liquidation": False,
            "margin_level": float('inf'),
            "episode": {"r": (self.balance / self.initial_balance - 1) * 100}
        }