# backtester.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forex_envNew import ForexEnv
from bnn_policy import BNNActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from typing import Dict, Any, Optional

class Backtester:
    """
    外汇交易模型回测系统
    
    功能:
    1. 加载训练好的模型和环境
    2. 在测试数据上运行完整的交易策略
    3. 记录详细的交易历史和资产变化
    4. 生成全面的性能指标和可视化图表
    """
    
    def __init__(
        self,
        model_path: str,
        vec_normalize_path: str,
        test_df: pd.DataFrame,
        initial_balance: float = 10000.0,
        max_position: float = 0.3,
        cost_ratio: float = 0.0001,
        uncertainty_scale: float = 0.5,
        n_samples: int = 50
    ):
        """
        初始化回测器
        
        Args:
            model_path: 训练好的模型路径 (.zip)
            vec_normalize_path: 向量化环境归一化参数路径 (.pkl)
            test_df: 测试数据集 (需包含特征)
            initial_balance: 初始资金
            max_position: 最大仓位比例
            cost_ratio: 交易成本比例
            uncertainty_scale: 不确定性归一化系数
            n_samples: 不确定性估计的采样次数
        """
        # 参数校验
        if test_df.empty:
            raise ValueError("测试数据集不能为空")
        if max_position <= 0 or max_position > 1:
            raise ValueError("max_position 必须在 (0, 1] 范围内")
        
        # 加载模型和归一化环境
        self.model = PPO.load(model_path)
        self.vec_env = VecNormalize.load(vec_normalize_path, DummyVecEnv([lambda: ForexEnv(test_df)]))
        self.vec_env.training = False  # 禁用训练模式
        
        # 创建独立的环境用于回测
        self.env = ForexEnv(
            df=test_df,
            initial_balance=initial_balance,
            max_position=max_position,
            cost_ratio=cost_ratio,
            uncertainty_scale=uncertainty_scale
        )
        
        # 回测参数
        self.n_samples = n_samples
        self.initial_balance = initial_balance
        
        # 结果存储
        self.history = []
        self.performance_metrics = {}
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        运行完整的回测过程
        
        Returns:
            包含回测结果和指标的字典
        """
        # 重置环境
        obs, _ = self.env.reset()
        self.history = []
        episode_return = 0
        
        # 初始状态记录
        self._record_step(
            action=0, 
            obs=obs, 
            reward=0,
            uncertainty=0,
            terminated=False
        )
        
        # 运行回测
        while not self.env.done:
            # 使用模型预测动作（包含不确定性估计）
            if isinstance(self.model.policy, BNNActorCriticPolicy):
                action, _ = self.model.predict(obs, deterministic=True)
                feature_mean, feature_std = self.model.policy.predict_with_uncertainty(obs, self.n_samples)
                uncertainty = feature_std.mean()  # 使用平均标准差作为不确定性度量
            else:
                action, _ = self.model.predict(obs, deterministic=True)
                uncertainty = 0.0
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(
                action, 
                uncertainty_std=uncertainty
            )
            
            # 记录步骤
            self._record_step(
                action=action,
                obs=next_obs,
                reward=reward,
                uncertainty=uncertainty,
                terminated=terminated or truncated
            )
            
            # 更新状态
            obs = next_obs
            episode_return += reward
        
        # 计算性能指标
        self._calculate_performance_metrics()
        
        return {
            "history": pd.DataFrame(self.history),
            "metrics": self.performance_metrics
        }
    
    def _record_step(
        self,
        action: int,
        obs: np.ndarray,
        reward: float,
        uncertainty: float,
        terminated: bool
    ):
        """记录每一步的回测信息"""
        self.history.append({
            "timestamp": self.env.t,
            "price": self.env.df.iloc[self.env.t]["close"],
            "action": action,
            "position_type": self.env.position_type,
            "position_size": self.env.position_size,
            "net_worth": self.env.net_worth,
            "balance": self.env.balance,
            "equity": self.env.equity,
            "reward": reward,
            "uncertainty": uncertainty,
            "RCI9": obs[0] * 100,  # 反归一化
            "RCI14": obs[1] * 100,
            "RCI21": obs[2] * 100,
            "bayes_prob": obs[3],
            "volatility": obs[4],
            "ma_ratio": obs[5] + 1,
            "terminated": terminated
        })
    
    def _calculate_performance_metrics(self):
        """计算回测性能指标"""
        df = pd.DataFrame(self.history)
        net_worth = df["net_worth"].values
        
        # 基本指标
        total_return = (net_worth[-1] / self.initial_balance - 1) * 100
        max_net_worth = np.maximum.accumulate(net_worth)
        drawdown = (max_net_worth - net_worth) / max_net_worth
        max_drawdown = np.max(drawdown) * 100
        
        # 交易指标
        trades = df[df["action"] != 0]
        winning_trades = trades[trades["reward"] > 0]
        losing_trades = trades[trades["reward"] < 0]
        
        # 风险调整收益
        returns = df["net_worth"].pct_change().dropna()
        sharpe_ratio = np.sqrt(252 * 60) * returns.mean() / (returns.std() + 1e-8)
        sortino_ratio = np.sqrt(252 * 60) * returns.mean() / (returns[returns < 0].std() + 1e-8)
        
        # 保存指标
        self.performance_metrics = {
            "initial_balance": self.initial_balance,
            "final_net_worth": net_worth[-1],
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "total_trades": len(trades),
            "win_rate": len(winning_trades) / len(trades) if len(trades) > 0 else 0,
            "avg_win": winning_trades["reward"].mean() if len(winning_trades) > 0 else 0,
            "avg_loss": losing_trades["reward"].mean() if len(losing_trades) > 0 else 0,
            "profit_factor": winning_trades["reward"].sum() / abs(losing_trades["reward"].sum()) if len(losing_trades) > 0 else float('inf'),
            "uncertainty_avg": df["uncertainty"].mean()
        }
    
    def generate_report(self, save_path: Optional[str] = None):
        """
        生成并保存回测报告和图表
        
        Args:
            save_path: 报告保存路径 (None 则只显示不保存)
        """
        if not self.history:
            raise RuntimeError("请先运行回测 (run_backtest)")
        
        df = pd.DataFrame(self.history)
        metrics = self.performance_metrics
        
        # 创建图表
        plt.figure(figsize=(15, 18))
        
        # 净值曲线
        plt.subplot(3, 1, 1)
        plt.plot(df["net_worth"], label="Net Worth")
        plt.plot(df["balance"], label="Balance", alpha=0.7)
        plt.title(f"Equity Curve (Final: ${metrics['final_net_worth']:,.2f})")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.grid(True)
        
        # 仓位和动作
        plt.subplot(3, 1, 2)
        plt.plot(df["position_type"], label="Position Type", color="purple")
        plt.bar(df.index, df["position_size"], alpha=0.3, label="Position Size", color="green")
        
        # 标记交易点
        buy_points = df[df["action"] == 1].index
        sell_points = df[df["action"] == 2].index
        plt.scatter(buy_points, df.loc[buy_points, "position_type"], 
                   marker="^", color="green", s=100, label="Buy")
        plt.scatter(sell_points, df.loc[sell_points, "position_type"], 
                   marker="v", color="red", s=100, label="Sell")
        
        plt.title("Position Management")
        plt.axhline(0, color="black", linestyle="--", alpha=0.3)
        plt.legend()
        plt.grid(True)
        
        # 技术指标和不确定性
        plt.subplot(3, 1, 3)
        plt.plot(df["RCI9"], label="RCI9", alpha=0.7)
        plt.plot(df["RCI14"], label="RCI14", alpha=0.7)
        plt.plot(df["RCI21"], label="RCI21", alpha=0.7)
        plt.plot(df["uncertainty"] * 100, label="Uncertainty (Scaled)", color="black", linestyle="--")
        plt.title("Technical Indicators and Model Uncertainty")
        plt.legend()
        plt.grid(True)
        
        # 添加指标表格
        metrics_text = (
            f"Total Return: {metrics['total_return_pct']:.2f}%\n"
            f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
            f"Total Trades: {metrics['total_trades']}\n"
            f"Win Rate: {metrics['win_rate']*100:.1f}%\n"
            f"Avg Win: {metrics['avg_win']*100:.3f}%\n"
            f"Avg Loss: {metrics['avg_loss']*100:.3f}%\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Avg Uncertainty: {metrics['uncertainty_avg']:.4f}"
        )
        plt.figtext(0.1, 0.02, metrics_text, bbox={"facecolor": "white", "alpha": 0.8}, 
                    fontsize=10, family="monospace")
        
        # 保存或显示
        if save_path:
            plt.savefig(f"{save_path}/backtest_report.png", bbox_inches="tight")
            print(f"✅ 报告已保存至: {save_path}/backtest_report.png")
        else:
            plt.tight_layout()
            plt.show()
        
        # 返回详细结果
        return {
            "history": df,
            "metrics": metrics
        }

# 使用示例
if __name__ == "__main__":
    # 1. 准备测试数据
    test_df = pd.read_csv('./data/2024min1.csv')
    test_df = test_df.iloc[15000:20000]  # 取500数据
    test_df.reset_index(inplace=True)

    from featuresNew import add_features
    test_df = add_features(test_df)
    test_df.dropna(inplace=True)
    
    # 2. 初始化回测器
    backtester = Backtester(
        model_path="bnn_ppo_forex_final.zip",
        vec_normalize_path="eval_vec_normalize.pkl",
        test_df=test_df,
        initial_balance=10000.0,
        max_position=0.3,
        n_samples=30
    )
    
    # 3. 运行回测
    results = backtester.run_backtest()
    
    # 4. 生成报告
    report = backtester.generate_report(save_path="./backtest_results")
    
    # 5. 打印关键指标
    print("\n" + "="*50)
    print("回测性能总结:")
    print("="*50)
    for k, v in report["metrics"].items():
        print(f"{k.replace('_', ' ').title():<20}: {v}")