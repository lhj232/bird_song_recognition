# -*- coding: utf-8 -*-
"""
mambabird/data/datasets/birdclef_dataset.py

BirdCLEF2024 数据集类定义模块。

【本次修复】
    真实BirdCLEF数据集中存在少量损坏、超短、非标准编码的音频文件，
    若不做容错处理，DataLoader worker一旦加载到坏文件会导致整个训练进程崩溃。
    本次修复为 __getitem__ 增加 try/except 容错：
        - 加载失败时记录warning日志，并自动改用同一数据集中的下一个样本重试，
          最多重试若干次后兜底返回静音波形，保证训练流程不会因个别坏文件中断。
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from mambabird.data.preprocessing.audio_cleaning import fix_length, load_and_resample
from mambabird.utils.logger import get_logger

logger = get_logger(__name__)

# 单个样本加载失败时的最大重试次数（改用其他样本重试，避免死循环）
_MAX_LOAD_RETRY = 3


class BirdCLEFDataset(Dataset):
    """
    BirdCLEF2024 数据集类。

    每个样本返回：
        waveform: 形状为 (num_samples,) 的定长单通道波形张量
        label_idx: 整数标签索引
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        audio_root_dir: Union[str, Path],
        filename_column: str,
        label_column: str,
        label_to_idx: Dict[str, int],
        sample_rate: int,
        clip_duration: float,
        is_train: bool = True,
    ) -> None:
        """
        参数：
            metadata_df: 已完成训练/验证/测试划分的元数据子集（DataFrame）。
            audio_root_dir: 音频文件根目录（即 raw_data_dir/audio_subdir）。
            filename_column: 元数据中音频文件名所在列名。
            label_column: 元数据中标签所在列名。
            label_to_idx: 标签字符串到整数索引的映射字典。
            sample_rate: 目标采样率（Hz）。
            clip_duration: 每段音频切片时长（秒）。
            is_train: 是否为训练集（控制定长裁剪时是否使用随机裁剪起点）。
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.audio_root_dir = Path(audio_root_dir)
        self.filename_column = filename_column
        self.label_column = label_column
        self.label_to_idx = label_to_idx
        self.sample_rate = sample_rate
        self.target_num_samples = int(sample_rate * clip_duration)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.metadata_df)

    def _load_single_sample(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        加载单个样本的原始逻辑（不含容错），供 __getitem__ 在try/except中调用。
        """
        row = self.metadata_df.iloc[index]

        audio_path = self.audio_root_dir / str(row[self.filename_column])
        label_str = str(row[self.label_column])
        label_idx = self.label_to_idx[label_str]

        waveform = load_and_resample(audio_path, target_sample_rate=self.sample_rate)
        waveform = fix_length(
            waveform,
            target_num_samples=self.target_num_samples,
            random_crop=self.is_train,
        )
        return waveform, label_idx

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        【容错版】加载样本。

        若指定index的音频文件加载失败（损坏/格式不支持/文件缺失等），
        记录warning日志并随机改用数据集中的另一个样本重试，
        重试 _MAX_LOAD_RETRY 次后仍失败，则兜底返回静音波形 + 该样本原始标签，
        保证训练/验证流程不会因个别坏文件而整体中断。
        """
        current_index = index

        for retry_count in range(_MAX_LOAD_RETRY):
            try:
                return self._load_single_sample(current_index)
            except Exception as exc:  # noqa: BLE001
                row = self.metadata_df.iloc[current_index]
                bad_filename = row[self.filename_column]
                logger.warning(
                    f"音频加载失败（第{retry_count + 1}次重试），"
                    f"文件: {bad_filename}，错误: {exc}"
                )
                # 改用下一个索引重试（循环取模，避免越界）
                current_index = (current_index + 1) % len(self.metadata_df)

        # 多次重试仍失败：兜底返回静音波形，保证不中断训练
        logger.error(
            f"样本index={index} 经{_MAX_LOAD_RETRY}次重试仍加载失败，"
            f"已使用静音波形兜底，请检查该文件是否损坏。"
        )
        fallback_label_str = str(self.metadata_df.iloc[index][self.label_column])
        fallback_label_idx = self.label_to_idx[fallback_label_str]
        silent_waveform = torch.zeros(self.target_num_samples)
        return silent_waveform, fallback_label_idx


def split_metadata(
    metadata_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    将元数据按比例随机划分为训练/验证/测试三个子集。

    参数：
        metadata_df: 完整元数据 DataFrame。
        train_ratio / val_ratio / test_ratio: 三者之和应为1.0（允许有少量浮点误差）。
        random_seed: 随机种子，保证划分结果可复现。

    返回：
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio 必须等于1.0"

    shuffled_df = metadata_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    num_samples = len(shuffled_df)
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    train_df = shuffled_df.iloc[:train_end]
    val_df = shuffled_df.iloc[train_end:val_end]
    test_df = shuffled_df.iloc[val_end:]

    return train_df, val_df, test_df