# -*- coding: utf-8 -*-
"""
mambabird/frontend/base_frontend.py

声学前端抽象基类模块。
所有前端实现（LogMel/LEAF/mCAF）必须继承该基类并实现 forward 方法，
保证训练器/骨干网络可以以统一接口调用任意前端，无需感知具体实现细节。
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseFrontend(nn.Module, ABC):
    """
    声学前端抽象基类。

    输入：
        waveform: 形状为 (batch_size, num_samples) 的原始波形张量
    输出：
        features: 形状为 (batch_size, channels, freq_bins, time_steps) 的特征张量
                  （与图像分类的 (B, C, H, W) 格式一致，便于直接接入CNN骨干网络）
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        前向计算：将原始波形转换为声学特征。

        参数：
            waveform: 形状为 (batch_size, num_samples) 的波形张量。

        返回：
            形状为 (batch_size, channels, freq_bins, time_steps) 的特征张量。
        """
        raise NotImplementedError