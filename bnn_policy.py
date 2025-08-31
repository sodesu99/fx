# bnn_policy.py - 修复版
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn

class MC_Dropout_FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(MC_Dropout_FeaturesExtractor, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, features_dim),
        )
        
    def forward(self, x):
        return self.net(x)

    def forward_with_uncertainty(self, x, n_samples=50):
        """
        ✅ 修复：正确的状态管理和设备处理
        """
        # 保存原始状态
        original_training = self.training
        device = next(self.parameters()).device
        
        # 设置为评估模式，但保持dropout为训练模式
        self.eval()
        for m in self.net.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        # 输入预处理
        if isinstance(x, np.ndarray):
            x = th.tensor(x, dtype=th.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(device)

        predictions = []
        with th.no_grad():
            for _ in range(n_samples):
                pred = self.net(x)
                predictions.append(pred.detach().cpu())
        
        predictions = th.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        # ✅ 修复：正确恢复原始状态
        self.train(original_training)
        if not original_training:
            for m in self.net.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()

        return mean.to(device), std.to(device)

class BNNActorCriticPolicy(ActorCriticPolicy):
    """
    ✅ 修复：增加错误处理和设备管理
    """
    def __init__(self, *args, **kwargs):
        super(BNNActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=MC_Dropout_FeaturesExtractor,
            features_extractor_kwargs={"features_dim": 256},
        )

    def forward(self, obs, deterministic=False):
        """标准前向流程"""
        return super(BNNActorCriticPolicy, self).forward(obs, deterministic=deterministic)

    def predict_with_uncertainty(self, obs, n_samples=50):
        """
        ✅ 修复：增加错误处理和类型检查
        """
        try:
            if isinstance(obs, np.ndarray):
                obs = th.tensor(obs, dtype=th.float32).to(self.device)
            elif isinstance(obs, th.Tensor):
                obs = obs.to(self.device)
            else:
                raise TypeError(f"不支持的观测类型: {type(obs)}")
                
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            # 使用特征提取器的不确定性方法
            feature_mean, feature_std = self.features_extractor.forward_with_uncertainty(obs, n_samples)
            
            return feature_mean, feature_std
            
        except Exception as e:
            print(f"⚠️ 不确定性预测失败: {e}")
            # 返回零不确定性作为后备
            dummy_features = th.zeros((obs.shape[0], 256), device=self.device)
            return dummy_features, dummy_features