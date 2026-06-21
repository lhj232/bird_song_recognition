# -*- coding: utf-8 -*-
"""
mambabird/backbone/registry.py

骨干网络注册器模块。
通过字符串类型名映射到具体骨干网络类，实现"配置驱动、无需修改训练代码即可切换骨干网络"的目标。
未来新增 EfficientNet / MobileNet / ViT / AST / Mamba 时，只需在 BACKBONE_REGISTRY 中追加映射关系。
"""

from typing import Any, Dict

from mambabird.backbone.base_backbone import BaseBackbone
from mambabird.backbone.cnn.resnet import ResNetBackbone

# 骨干网络类型名 -> 骨干网络类 的注册表
BACKBONE_REGISTRY = {
    "resnet18": ResNetBackbone,
    "resnet34": ResNetBackbone,
    "resnet50": ResNetBackbone,
    # "efficientnet": EfficientNetBackbone,   # 预留：未来接入EfficientNet
    # "mobilenet": MobileNetBackbone,         # 预留：未来接入MobileNet
    # "vit": ViTBackbone,                     # 预留：未来接入ViT
    # "ast": ASTBackbone,                     # 预留：未来接入AST
    # "mamba": MambaBackbone,                 # 预留：未来接入Mamba
}


def build_backbone(backbone_config: Dict[str, Any]) -> BaseBackbone:
    """
    根据配置构建对应的骨干网络实例。

    参数：
        backbone_config: 来自实验YAML中 "backbone" 字段的配置字典，必须包含 "type" 字段。

    返回：
        对应的 BaseBackbone 子类实例。

    异常：
        ValueError: 当 backbone_config["type"] 不在注册表中时抛出。
    """
    backbone_type = backbone_config["type"]

    if backbone_type not in BACKBONE_REGISTRY:
        raise ValueError(
            f"未知的骨干网络类型: {backbone_type}，"
            f"当前已注册骨干网络类型: {list(BACKBONE_REGISTRY.keys())}"
        )

    backbone_cls = BACKBONE_REGISTRY[backbone_type]
    return backbone_cls(backbone_config=backbone_config)