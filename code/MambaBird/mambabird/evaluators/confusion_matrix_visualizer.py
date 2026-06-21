# -*- coding: utf-8 -*-
"""
mambabird/evaluators/confusion_matrix_visualizer.py

混淆矩阵可视化模块。
负责将 ClassificationEvaluator 计算出的混淆矩阵渲染为图像，
支持保存为本地图片文件，以及直接写入TensorBoard，
便于在类别数较多（鸟声识别通常上百个物种）的场景下，
通过归一化与降采样标签等方式保证图像可读性。
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")  # 服务器/无显示环境下渲染图像，避免GUI依赖
import matplotlib.pyplot as plt
import numpy as np


def normalize_confusion_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    """
    按行（真实类别）归一化混淆矩阵，每行之和为1，
    便于在类别样本数不均衡时直观比较各类别的识别准确率分布。

    参数：
        confusion_matrix: 原始混淆矩阵，形状为 (num_classes, num_classes)。

    返回：
        归一化后的混淆矩阵（同形状，元素为浮点数）。
    """
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    # 避免除零：某个真实类别在本次评估样本中未出现时，行和为0
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    normalized = confusion_matrix.astype(np.float64) / row_sums_safe
    return normalized


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    label_names: Optional[list] = None,
    normalize: bool = True,
    max_labels_to_show: int = 30,
    figure_title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    绘制混淆矩阵热力图。

    参数：
        confusion_matrix: 混淆矩阵，形状为 (num_classes, num_classes)。
        label_names: 类别名称列表（按索引顺序），用于坐标轴标注。
        normalize: 是否按行归一化后再绘制（鸟声识别类别数通常较多，
                   归一化后更便于观察每个类别自身的识别准确率）。
        max_labels_to_show: 当类别数超过该值时，坐标轴不再逐一显示类别名称
                            （避免上百个物种名称重叠不可读），仅显示矩阵本身。
        figure_title: 图像标题。
        save_path: 若指定，将图像保存至该路径（自动创建父目录）。

    返回：
        matplotlib Figure 对象（同时已根据save_path完成保存，若指定）。
    """
    matrix_to_plot = (
        normalize_confusion_matrix(confusion_matrix) if normalize else confusion_matrix
    )

    num_classes = matrix_to_plot.shape[0]
    # 类别数较多时适当放大图像尺寸，保证矩阵主体清晰
    fig_size = max(6, min(20, num_classes * 0.3))

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix_to_plot, interpolation="nearest", cmap="Blues")
    ax.set_title(figure_title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    if label_names is not None and num_classes <= max_labels_to_show:
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(label_names, rotation=90, fontsize=6)
        ax.set_yticklabels(label_names, fontsize=6)
    else:
        # 类别过多时不显示逐类别标签，仅保留坐标轴范围信息
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def confusion_matrix_to_tensorboard_image(
    confusion_matrix: np.ndarray,
    label_names: Optional[list] = None,
    normalize: bool = True,
    max_labels_to_show: int = 30,
    figure_title: str = "Confusion Matrix",
) -> np.ndarray:
    """
    将混淆矩阵渲染为图像数组，格式为 (channels, height, width)，
    可直接通过 TensorBoardLogger 的 SummaryWriter.add_image 接口写入TensorBoard。

    参数：
        与 plot_confusion_matrix 相同（除 save_path）。

    返回：
        形状为 (3, H, W) 的 RGB 图像数组（取值范围 [0, 1]），符合
        torch.utils.tensorboard.SummaryWriter.add_image 默认的 CHW 格式要求。
    """
    fig = plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        label_names=label_names,
        normalize=normalize,
        max_labels_to_show=max_labels_to_show,
        figure_title=figure_title,
        save_path=None,
    )

    fig.canvas.draw()
    # 从matplotlib画布读取RGBA像素，转换为 (H, W, 3) 后再转为 (3, H, W)
    image_rgba = np.asarray(fig.canvas.buffer_rgba())
    image_rgb = image_rgba[:, :, :3]
    image_chw = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32) / 255.0

    plt.close(fig)

    return image_chw