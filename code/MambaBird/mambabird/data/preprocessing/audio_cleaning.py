# -*- coding: utf-8 -*-
"""
mambabird/data/preprocessing/audio_cleaning.py

音频预处理模块。
负责音频文件的加载、重采样、单通道转换以及定长裁剪/补零，
为后续声学前端（LogMel/mCAF等）提供统一长度的原始波形输入。
"""

from pathlib import Path
from typing import Union

import torch
import torchaudio


def load_and_resample(
    audio_path: Union[str, Path],
    target_sample_rate: int,
) -> torch.Tensor:
    """
    加载音频文件并重采样至目标采样率，同时转换为单通道。

    参数：
        audio_path: 音频文件路径。
        target_sample_rate: 目标采样率（Hz）。

    返回：
        形状为 (num_samples,) 的一维波形张量。
    """
    waveform, original_sample_rate = torchaudio.load(str(audio_path))

    # 多通道转单通道：取各通道均值
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 重采样到目标采样率
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # (num_samples,)


def fix_length(
    waveform: torch.Tensor,
    target_num_samples: int,
    random_crop: bool = True,
) -> torch.Tensor:
    """
    将波形裁剪或补零至固定长度，保证一个batch内所有样本长度一致。

    参数：
        waveform: 形状为 (num_samples,) 的一维波形张量。
        target_num_samples: 目标采样点数。
        random_crop: 当波形长度超过目标长度时，是否随机裁剪起点
                     （训练阶段建议True以增加多样性，验证/测试阶段建议False取开头）。

    返回：
        形状为 (target_num_samples,) 的定长波形张量。
    """
    current_length = waveform.shape[0]

    if current_length == target_num_samples:
        return waveform

    if current_length > target_num_samples:
        if random_crop:
            max_start = current_length - target_num_samples
            start = torch.randint(low=0, high=max_start + 1, size=(1,)).item()
        else:
            start = 0
        return waveform[start: start + target_num_samples]

    # current_length < target_num_samples：末尾补零
    padding_length = target_num_samples - current_length
    return torch.nn.functional.pad(waveform, (0, padding_length), mode="constant", value=0.0)