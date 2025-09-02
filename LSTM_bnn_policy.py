import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

class LSTM_BNN_FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, lstm_hidden_size=128, n_lstm_layers=1):
        super(LSTM_BNN_FeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # 对于2D观察空间 (seq_len, features)，取特征维度
        if len(observation_space.shape) == 2:
            self.input_dim = observation_space.shape[1]  # 特征维度
        else:
            self.input_dim = observation_space.shape[0]  # 每步输入维度
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        
        # LSTM 主干（我们会在推理时启用 dropout）
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=0.3 if n_lstm_layers > 1 else 0
        )
        
        # Dropout 层（手动控制）
        self.dropout = nn.Dropout(0.3)
        
        # 输出映射到特征空间
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, features_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
        if x.dim() == 2:
            # 如果没有时间维度，假设 seq_len=1
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        elif x.dim() == 3:
            # 如果已经是3D，确保维度顺序正确
            # x shape: (batch_size, seq_len, input_dim)
            pass

        lstm_out, (hidden, _) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden)
        last_hidden = hidden[-1]  # 取最后一层的最后时刻 hidden state

        # 应用 dropout（训练和推理都开启）
        dropped = self.dropout(last_hidden)
        features = self.fc(dropped)
        return features

    def forward_with_uncertainty(self, x, n_samples=50):
        """
        使用 MC Dropout 估计不确定性
        """
        # 保存原始状态
        original_training = self.training
        device = next(self.parameters()).device

        # 设置为 eval 模式，但强制 dropout 开启
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.LSTM)):
                m.train()  # 强制开启 dropout

        if isinstance(x, np.ndarray):
            x = th.tensor(x, dtype=th.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)  # (1, 1, input_dim) 或 (1, T, input_dim)
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # (1, T, input_dim)
        x = x.to(device)

        predictions = []
        with th.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred.detach().cpu())
        
        predictions = th.stack(predictions)  # (n_samples, batch, features_dim)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        # 恢复原始状态
        self.train(original_training)
        if not original_training:
            for m in self.modules():
                if isinstance(m, (nn.Dropout, nn.LSTM)):
                    m.eval()

        return mean.to(device), std.to(device)

# 使用 SACPolicy 而不是 ActorCriticPolicy
class LSTM_BNN_Policy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(LSTM_BNN_Policy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=LSTM_BNN_FeaturesExtractor,
            features_extractor_kwargs={"features_dim": 256, "lstm_hidden_size": 128},
        )

    def predict_with_uncertainty(self, obs, n_samples=50):
        try:
            if isinstance(obs, np.ndarray):
                obs = th.tensor(obs, dtype=th.float32).to(self.device)
            elif isinstance(obs, th.Tensor):
                obs = obs.to(self.device)
            else:
                raise TypeError(f"不支持的观测类型: {type(obs)}")

            if obs.dim() == 1:
                obs = obs.unsqueeze(0)  # (1, input_dim)
            elif obs.dim() == 2 and obs.size(0) != 1:
                obs = obs.unsqueeze(0)  # (1, T, input_dim)

            return self.features_extractor.forward_with_uncertainty(obs, n_samples)
        except Exception as e:
            print(f"⚠️ 不确定性预测失败: {e}")
            dummy = th.zeros((1, 256), device=self.device)
            return dummy, dummy