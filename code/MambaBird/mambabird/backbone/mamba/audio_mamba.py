# -*- coding: utf-8 -*-
"""
mambabird/backbone/mamba/audio_mamba.py

AudioMamba 骨干网络实现模块（V2核心新增）。

设计说明：
    AudioMamba 严格遵循项目统一的 BaseBackbone 接口：
        输入: features，形状为 (batch_size, channels, freq_bins, time_steps)
              —— 与LogMel/mCAF等任意前端输出格式完全一致
        输出: logits，形状为 (batch_size, num_classes)

    因此 AudioMamba 可以与现有 BirdSoundClassifier（前端+骨干网络组合）、
    BaseTrainer（训练循环）、ClassificationEvaluator（评估器）无缝拼接，
    无需对这三者做任何修改，仅需在骨干网络注册器中追加映射关系。

    内部处理逻辑：
        1. 将声学特征 (B, C, n_mels, T) 转换为序列形式 (B, T, n_mels*C)，
           把"时间"作为序列维度，"频率(梅尔通道)"作为每个时间步的特征向量，
           这是将Mamba（专为序列建模设计）应用于频谱图的标准做法。
        2. 经线性层投影到 d_model 维度。
        3. 堆叠 n_layers 个 SelectiveSSMBlock，沿时间维度建模长程依赖
           （Mamba的核心优势：线性时间复杂度处理长序列，适合鸟鸣等长时序信号）。
        4. 对所有时间步做平均池化（Mean Pooling），得到整段音频的全局表征。
        5. 经分类头线性层输出 num_classes 维logits。
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from mambabird.backbone.base_backbone import BaseBackbone
from mambabird.backbone.mamba.selective_ssm import SelectiveSSMBlock


class AudioMamba(BaseBackbone):
    """
    基于选择性状态空间模型（Mamba）的鸟声分类骨干网络。
    """

    def __init__(self, backbone_config: Dict[str, Any]) -> None:
        """
        参数：
            backbone_config: 来自实验YAML中 "backbone" 字段的配置字典，需包含：
                d_model: 序列建模的隐藏维度
                n_layers: 堆叠的SelectiveSSMBlock层数
                d_state: SSM隐状态维度
                d_conv: 因果卷积核大小
                expand_factor: 内部升维倍数
                num_classes: 分类类别数
                in_channels: 输入特征通道数（默认1，对应LogMel单通道）
        """
        super().__init__()

        self.d_model = backbone_config["d_model"]
        n_layers = backbone_config["n_layers"]
        d_state = backbone_config.get("d_state", 16)
        d_conv = backbone_config.get("d_conv", 4)
        expand_factor = backbone_config.get("expand_factor", 2)
        num_classes = backbone_config["num_classes"]
        self.in_channels = backbone_config.get("in_channels", 1)

        # 输入投影层：在forward中根据实际频率维度(n_mels * in_channels)动态构建，
        # 因为不同前端配置（n_mels不同）会导致输入特征维度不同，此处先声明为None，
        # 延迟到第一次forward时根据实际输入形状惰性构建（Lazy Initialization）。
        self._input_proj: nn.Module = None
        self._expected_feature_dim: int = None

        # Mamba主干：堆叠多个SelectiveSSMBlock
        self.mamba_layers = nn.ModuleList(
            [
                SelectiveSSMBlock(
                    d_model=self.d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand_factor=expand_factor,
                )
                for _ in range(n_layers)
            ]
        )

        # 层归一化：在每层Mamba Block之后做归一化，稳定深层堆叠的训练
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(n_layers)]
        )

        # 最终归一化 + 分类头
        self.final_norm = nn.LayerNorm(self.d_model)
        self.classifier_head = nn.Linear(self.d_model, num_classes)

    def _build_input_projection_if_needed(self, feature_dim: int, device: torch.device) -> None:
        """
        惰性构建输入投影层：首次forward时根据实际输入的频率维度构建Linear层。
        后续若输入维度发生变化（理论上不应该，因为前端配置固定），会重新构建并报警。
        """
        if self._input_proj is None or self._expected_feature_dim != feature_dim:
            self._input_proj = nn.Linear(feature_dim, self.d_model).to(device)
            self._expected_feature_dim = feature_dim
            # 将新构建的层注册为子模块，保证其参数能被优化器正确捕获
            self.add_module("input_proj", self._input_proj)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        参数：
            features: 形状为 (batch_size, channels, freq_bins, time_steps) 的声学特征张量
                      （与LogMel/mCAF等任意前端输出格式一致）。

        返回：
            形状为 (batch_size, num_classes) 的logits张量。
        """
        batch_size, channels, freq_bins, time_steps = features.shape

        # 1. 将 (B, C, F, T) 转换为序列形式 (B, T, C*F)
        #    时间维度作为序列长度，通道与频率维度合并作为每个时间步的特征向量
        x = features.permute(0, 3, 1, 2).contiguous()  # (B, T, C, F)
        x = x.view(batch_size, time_steps, channels * freq_bins)  # (B, T, C*F)

        # 2. 惰性构建并应用输入投影层，将特征维度统一映射到 d_model
        self._build_input_projection_if_needed(
            feature_dim=channels * freq_bins, device=features.device
        )
        x = self._input_proj(x)  # (B, T, d_model)

        # 3. 依次通过堆叠的Mamba Block（每层后接LayerNorm，类似Transformer的Pre-Norm结构）
        for mamba_block, norm in zip(self.mamba_layers, self.layer_norms):
            x = mamba_block(x)
            x = norm(x)

        x = self.final_norm(x)

        # 4. 时间维度平均池化，得到整段音频的全局表征 (B, d_model)
        pooled = x.mean(dim=1)

        # 5. 分类头输出logits (B, num_classes)
        logits = self.classifier_head(pooled)
        return logits