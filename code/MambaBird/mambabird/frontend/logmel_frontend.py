# -*- coding: utf-8 -*-
"""
mambabird/frontend/logmel_frontend.py

LogMel 静态声学前端实现模块（V1基线前端）。
使用 torchaudio 的 MelSpectrogram + AmplitudeToDB 计算对数梅尔频谱图，
作为后续 mCAF 等可学习前端的对照基线。
"""

from typing import Any, Dict

import torch
import torchaudio

from mambabird.frontend.base_frontend import BaseFrontend


class LogMelFrontend(BaseFrontend):
    """
    LogMel 前端：waveform -> Mel频谱图 -> 对数变换。
    """

    def __init__(self, frontend_config: Dict[str, Any], sample_rate: int) -> None:
        """
        参数：
            frontend_config: 来自实验YAML中 "frontend" 字段的配置字典，需包含：
                n_mels, n_fft, hop_length, win_length, fmin, fmax
            sample_rate: 音频采样率（Hz），需与数据配置中的sample_rate保持一致。
        """
        super().__init__()

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=frontend_config["n_fft"],
            win_length=frontend_config["win_length"],
            hop_length=frontend_config["hop_length"],
            f_min=frontend_config["fmin"],
            f_max=frontend_config["fmax"],
            n_mels=frontend_config["n_mels"],
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        参数：
            waveform: 形状为 (batch_size, num_samples) 的波形张量。

        返回：
            形状为 (batch_size, 1, n_mels, time_steps) 的对数梅尔频谱特征。
        """
        # MelSpectrogram 输出形状: (batch_size, n_mels, time_steps)
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        # 增加单通道维度，统一为 (batch_size, channels=1, n_mels, time_steps)
        log_mel_spec = log_mel_spec.unsqueeze(1)

        return log_mel_spec