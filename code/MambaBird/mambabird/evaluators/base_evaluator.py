# -*- coding: utf-8 -*-
"""
mambabird/evaluators/base_evaluator.py

评估器抽象基类模块。
定义统一的评估器接口，未来若新增"多模态评估器"（结合元数据/拓扑指标等）
或"边缘部署评估器"（关注延迟/内存而非分类指标）时，可继承该基类，
保证调用方（训练脚本/测试脚本）以统一接口使用任意评估器。
"""

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

from mambabird.evaluators.evaluation_result import EvaluationResult


class BaseEvaluator(ABC):
    """
    评估器抽象基类。
    """

    @abstractmethod
    def evaluate(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        label_names: Optional[Iterable[str]] = None,
    ) -> EvaluationResult:
        """
        执行评估并返回结构化结果。

        参数：
            y_true: 真实标签索引序列。
            y_pred: 预测标签索引序列。
            label_names: 可选，类别索引到标签名称的列表（按索引顺序排列）。

        返回：
            EvaluationResult 实例。
        """
        raise NotImplementedError