# -*- coding: utf-8 -*-
"""
mambabird/backbone/cnn/resnet.py

ResNet 骨干网络实现模块（V1基线骨干网络）。
基于 torchvision 的 ResNet 结构，针对单通道声学特征输入（LogMel等）
对第一层卷积进行适配，并替换分类头为指定类别数。
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision.models as tv_models

from mambabird.backbone.base_backbone import BaseBackbone

# variant名称 -> torchvision构建函数 的映射
_RESNET_BUILDERS = {
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
}


class ResNetBackbone(BaseBackbone):
    """
    ResNet 骨干网络封装类。
    """

    def __init__(self, backbone_config: Dict[str, Any]) -> None:
        """
        参数：
            backbone_config: 来自实验YAML中 "backbone" 字段的配置字典，需包含：
                type（用于variant选择的兼容字段，实际variant由"variant"或"type"字段指定）,
                pretrained, in_channels, num_classes
        """
        super().__init__()

        variant = backbone_config.get("variant", backbone_config.get("type", "resnet18"))
        pretrained = backbone_config.get("pretrained", False)
        in_channels = backbone_config.get("in_channels", 1)
        num_classes = backbone_config["num_classes"]

        if variant not in _RESNET_BUILDERS:
            raise ValueError(
                f"未知的ResNet变体: {variant}，"
                f"当前支持: {list(_RESNET_BUILDERS.keys())}"
            )

        # 构建标准torchvision ResNet（不加载预训练权重时weights=None）
        builder_fn = _RESNET_BUILDERS[variant]
        weights = "DEFAULT" if pretrained else None
        self.resnet = builder_fn(weights=weights)

        # 适配单通道（或其他通道数）输入：替换第一层卷积
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None),
        )

        # 替换分类头为指定类别数
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        参数：
            features: 形状为 (batch_size, in_channels, freq_bins, time_steps) 的特征张量。

        返回：
            形状为 (batch_size, num_classes) 的logits张量。
        """
        return self.resnet(features)