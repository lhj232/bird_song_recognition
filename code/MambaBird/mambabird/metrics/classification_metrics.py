# -*- coding: utf-8 -*-
"""
mambabird/metrics/classification_metrics.py

分类评估指标计算模块。
统一计算准确率（Accuracy）与F1分数（宏平均），
保证全项目所有实验的指标计算口径一致，便于论文表格直接引用。
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """
    计算分类任务的准确率与宏平均F1分数。

    参数：
        y_true: 真实标签索引列表。
        y_pred: 预测标签索引列表。

    返回：
        包含 "accuracy" 与 "f1_macro" 两个键的指标字典。
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    f1_macro = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
    }