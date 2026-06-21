# -*- coding: utf-8 -*-
"""
mambabird/data/preprocessing/label_parser.py

标签解析模块。
负责从 BirdCLEF2024 官方元数据 CSV 中解析出全部物种标签，
构建 label <-> index 的双向映射，供数据集与评估模块统一使用。
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd


def build_label_mapping(
    metadata_csv: Union[str, Path],
    label_column: str = "primary_label",
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    从元数据 CSV 中解析全部唯一标签，构建标签到索引的双向映射。

    参数：
        metadata_csv: BirdCLEF2024 官方元数据 CSV 文件路径。
        label_column: 元数据中存放物种标签的列名（默认 "primary_label"）。

    返回：
        label_to_idx: {标签字符串: 整数索引} 的字典，按字母顺序排序，
                      保证多次运行映射结果一致（可复现性要求）。
        idx_to_label: {整数索引: 标签字符串} 的字典，为 label_to_idx 的逆映射。
    """
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"元数据文件不存在: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    if label_column not in df.columns:
        raise KeyError(f"元数据中未找到标签列: {label_column}")

    # 排序保证标签到索引的映射在多次运行中保持一致
    unique_labels: List[str] = sorted(df[label_column].unique().tolist())

    label_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label: Dict[int, str] = {idx: label for label, idx in label_to_idx.items()}

    return label_to_idx, idx_to_label