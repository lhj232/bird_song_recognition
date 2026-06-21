# -*- coding: utf-8 -*-
"""
mambabird/evaluators/

评估模块包。
负责对多分类鸟声识别模型进行统一、可复用的评估，包括：
    Accuracy / Precision / Recall / F1（micro/macro/weighted/per-class）/ Confusion Matrix。

该模块独立于 mambabird/metrics/（训练循环中实时记录的轻量指标），
evaluators/ 面向"完整评估流程"（如测试集最终评估、消融实验报告生成），
输出更全面的结构化评估结果，并支持混淆矩阵可视化与TensorBoard记录。
"""

from mambabird.evaluators.classification_evaluator import ClassificationEvaluator
from mambabird.evaluators.evaluation_result import EvaluationResult

__all__ = ["ClassificationEvaluator", "EvaluationResult"]