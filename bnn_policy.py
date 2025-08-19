# bnn_policy.py
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_schedule_fn

class MC_Dropout_FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):  # 改为256
        super(MC_Dropout_FeaturesExtractor, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, features_dim),  # 输出256维
        )
        
    def forward(self, x):
        return self.net(x)

    def forward_with_uncertainty(self, x, n_samples=50):
        # 保存原始状态
        training = self.training
        device = next(self.parameters()).device
        self.eval()  # 保持 eval 模式
        for m in self.net.modules():
            if isinstance(m, nn.Dropout):
                m.train()  # 仅启用 dropout

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

        # 恢复原始状态
        if not training:
            self.train()
            for m in self.net.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()


        return mean, std

class BNNActorCriticPolicy(ActorCriticPolicy):
    """
    正确实现：只替换 features_extractor，保留 action_net 和 value_net
    """
    def __init__(self, *args, **kwargs):
        super(BNNActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
            # net_arch 自动构建 mlp_extractor
            # net_arch=[64, 64],  # 提供给 mlp_extractor
            features_extractor_class=MC_Dropout_FeaturesExtractor,
            features_extractor_kwargs={"features_dim": 256},
        )
        # ✅ 不手动置空任何组件！

    def forward(self, obs, deterministic=False):
        """
        标准前向流程，让 SB3 自己处理
        """
        return super(BNNActorCriticPolicy, self).forward(obs, deterministic=deterministic)

    def predict_with_uncertainty(self, obs, n_samples=50):
        """
        新增方法：获取特征的不确定性
        """
        if isinstance(obs, np.ndarray):
            obs = th.tensor(obs, dtype=th.float32).to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # print("Features Extractor:", self.features_extractor)

        # 使用特征提取器的 uncertainty 方法
        feature_mean, feature_std = self.features_extractor.forward_with_uncertainty(obs, n_samples)
        # 将特征传入后续网络（mlp_extractor → action_net）
        # 注意：这里我们只关心特征不确定性，动作由标准流程决定
        return feature_mean, feature_std