# -*- coding: utf-8 -*-
"""
mambabird/data/dataloader_builder.py

统一 DataLoader 构建入口模块。

【本次修复】
    1. train_loader 增加 drop_last=True：
       避免最后一个batch size=1时，ResNet中的BatchNorm2d在训练模式下
       因"batch_size=1无法计算batch统计量"而抛出运行时错误。
    2. pin_memory 根据CUDA是否可用自动判定：
       CPU环境下强制关闭pin_memory，避免无意义的警告与内存占用。
    3. num_workers>0时开启 persistent_workers=True：
       避免每个epoch重新创建worker进程，在Windows(spawn模式)下显著提速。
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mambabird.data.datasets.birdclef_dataset import BirdCLEFDataset, split_metadata
from mambabird.data.preprocessing.label_parser import build_label_mapping


def build_dataloaders(
    data_config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """
    根据数据配置字典，构建训练/验证/测试三个 DataLoader。

    参数：
        data_config: 来自实验YAML中 "data" 字段的配置字典，需包含：
            raw_data_dir, metadata_csv, audio_subdir,
            filename_column, label_column,
            sample_rate, clip_duration,
            split: {train_ratio, val_ratio, test_ratio, random_seed},
            dataloader: {batch_size, num_workers, shuffle_train, pin_memory}

    返回：
        train_loader, val_loader, test_loader: 三个 DataLoader
        label_to_idx, idx_to_label: 标签映射字典（用于模型类别数与结果可解释性）
    """
    raw_data_dir = Path(data_config["raw_data_dir"])
    metadata_csv = Path(data_config["metadata_csv"])
    audio_root_dir = raw_data_dir / data_config["audio_subdir"]

    filename_column = data_config["filename_column"]
    label_column = data_config["label_column"]

    sample_rate = data_config["sample_rate"]
    clip_duration = data_config["clip_duration"]

    split_cfg = data_config["split"]
    loader_cfg = data_config["dataloader"]

    # ---------------- CUDA可用性检测，用于自动修正pin_memory ----------------
    cuda_available = torch.cuda.is_available()
    # 用户配置中开启了pin_memory，但当前环境无GPU，则自动关闭，避免警告与无效内存占用
    effective_pin_memory = bool(loader_cfg.get("pin_memory", True)) and cuda_available

    num_workers = int(loader_cfg.get("num_workers", 4))
    # num_workers=0时不支持persistent_workers，需置为False
    use_persistent_workers = num_workers > 0

    # 1. 构建标签映射
    label_to_idx, idx_to_label = build_label_mapping(
        metadata_csv=metadata_csv,
        label_column=label_column,
    )

    # 2. 加载元数据并划分训练/验证/测试集
    metadata_df = pd.read_csv(metadata_csv)
    train_df, val_df, test_df = split_metadata(
        metadata_df=metadata_df,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        random_seed=split_cfg["random_seed"],
    )

    # 3. 构建Dataset
    common_kwargs = dict(
        audio_root_dir=audio_root_dir,
        filename_column=filename_column,
        label_column=label_column,
        label_to_idx=label_to_idx,
        sample_rate=sample_rate,
        clip_duration=clip_duration,
    )

    train_dataset = BirdCLEFDataset(metadata_df=train_df, is_train=True, **common_kwargs)
    val_dataset = BirdCLEFDataset(metadata_df=val_df, is_train=False, **common_kwargs)
    test_dataset = BirdCLEFDataset(metadata_df=test_df, is_train=False, **common_kwargs)

    # 4. 封装DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=loader_cfg.get("shuffle_train", True),
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=use_persistent_workers,
        # 【修复】drop_last=True：避免最后一个batch size=1导致BatchNorm训练模式报错
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=use_persistent_workers,
        # 验证集无需drop_last：eval模式下BatchNorm使用running статистика，batch_size=1不会报错，
        # 且drop样本会导致评估指标不准确（漏评估部分样本）
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=use_persistent_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, label_to_idx, idx_to_label