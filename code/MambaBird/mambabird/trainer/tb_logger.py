# -*- coding: utf-8 -*-
"""
mambabird/trainer/tb_logger.py

TensorBoard 统一记录接口模块。

【本次修复】
    增加 log_image() 方法，封装 SummaryWriter.add_image，
    用于将混淆矩阵等可视化结果写入TensorBoard
    （此前 confusion_matrix_to_tensorboard_image 生成的图像数组无处可用，现已补齐接口）。
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    TensorBoard 日志记录器封装类。
    """

    def __init__(self, log_dir: Union[str, Path]) -> None:
        """
        参数：
            log_dir: TensorBoard 日志输出目录（通常为 experiments/<exp_name>/tb_logs）。
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """记录单个标量（如loss、accuracy等）。"""
        self.writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """批量记录多个标量。"""
        for tag, value in tag_scalar_dict.items():
            self.writer.add_scalar(tag, value, global_step=step)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """记录文本信息（如配置摘要、实验备注等）。"""
        self.writer.add_text(tag, text, global_step=step)

    def log_image(self, tag: str, image_chw: np.ndarray, step: int = 0) -> None:
        """
        记录单张图像（如混淆矩阵热力图、潜空间可视化等）。

        参数：
            tag: TensorBoard中显示的标签名称。
            image_chw: 形状为 (channels, height, width) 的图像数组，
                       取值范围应为 [0, 1]（与 SummaryWriter.add_image 默认要求一致）。
            step: 当前记录对应的step（如epoch编号）。
        """
        self.writer.add_image(tag, image_chw, global_step=step)

    def close(self) -> None:
        """关闭 SummaryWriter，释放资源。训练结束时必须调用。"""
        self.writer.flush()
        self.writer.close()