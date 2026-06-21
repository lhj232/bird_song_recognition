# -*- coding: utf-8 -*-
"""
mambabird/backbone/base_backbone.py

骨干网络抽象基类模块。
所有骨干网络实现（ResNet/EfficientNet/MobileNet/ViT/AST/Mamba）
必须继承该基类并实现 forward 方法，保证训练器可以统一接口调用任意骨干网络。
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """
    骨干网络抽象基类。

    输入：
        features: 形状为 (batch_size, channels, freq_bins, time_steps) 的声学特征张量
    输出：
        logits: 形状为 (batch_size, num_classes) 的分类logits张量
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向计算：将声学特征转换为分类logits。

        参数：
            features: 形状为 (batch_size, channels, freq_bins, time_steps) 的特征张量。

        返回：
            形状为 (batch_size, num_classes) 的logits张量。
        """
        raise NotImplementedError