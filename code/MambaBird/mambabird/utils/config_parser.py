# -*- coding: utf-8 -*-
"""
mambabird/utils/config_parser.py

YAML 配置解析模块。
负责将实验配置文件（YAML）加载为 Python 字典，并提供保存配置快照的功能，
是项目"配置驱动、实验可复现"原则的基础设施。
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载单个 YAML 配置文件为字典。

    参数：
        config_path: YAML 配置文件路径。

    返回：
        解析后的配置字典。

    异常：
        FileNotFoundError: 当配置文件不存在时抛出。
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return config_dict if config_dict is not None else {}


def save_yaml_config(config_dict: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    将配置字典保存为 YAML 文件（用于实验目录下的 config_snapshot.yaml）。

    参数：
        config_dict: 需要保存的配置字典。
        save_path: 保存路径。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False)