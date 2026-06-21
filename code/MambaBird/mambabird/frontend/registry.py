# -*- coding: utf-8 -*-
"""
mambabird/frontend/registry.py

声学前端注册器模块。
通过字符串类型名映射到具体前端类，实现"配置驱动、无需修改训练代码即可切换前端"的目标。
未来新增 LEAF / mCAF 等前端时，只需在 FRONTEND_REGISTRY 中追加映射关系。
"""

from typing import Any, Dict

from mambabird.frontend.base_frontend import BaseFrontend
from mambabird.frontend.logmel_frontend import LogMelFrontend

# 前端类型名 -> 前端类 的注册表
FRONTEND_REGISTRY = {
    "logmel": LogMelFrontend,
    # "leaf": LeafFrontend,      # 预留：未来接入LEAF前端
    # "mcaf": MCAFFrontend,      # 预留：未来接入mCAF前端
}


def build_frontend(frontend_config: Dict[str, Any], sample_rate: int) -> BaseFrontend:
    """
    根据配置构建对应的声学前端实例。

    参数：
        frontend_config: 来自实验YAML中 "frontend" 字段的配置字典，必须包含 "type" 字段。
        sample_rate: 音频采样率（Hz）。

    返回：
        对应的 BaseFrontend 子类实例。

    异常：
        ValueError: 当 frontend_config["type"] 不在注册表中时抛出。
    """
    frontend_type = frontend_config["type"]

    if frontend_type not in FRONTEND_REGISTRY:
        raise ValueError(
            f"未知的前端类型: {frontend_type}，"
            f"当前已注册前端类型: {list(FRONTEND_REGISTRY.keys())}"
        )

    frontend_cls = FRONTEND_REGISTRY[frontend_type]
    return frontend_cls(frontend_config=frontend_config, sample_rate=sample_rate)