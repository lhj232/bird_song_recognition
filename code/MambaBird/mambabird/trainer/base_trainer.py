# -*- coding: utf-8 -*-
"""
mambabird/trainer/base_trainer.py

标准训练器模块（Layer 4 核心实现）。

【本次修复】
    1. 类型标注兼容性修复：
       原代码使用 `torch.optim.lr_scheduler._LRScheduler` 作为类型标注，
       该属性为私有API，在不同PyTorch版本中可能被重命名/移除，
       由于Python默认会在函数定义时立即求值标注（未使用 from __future__ import annotations），
       一旦该属性不存在，整个模块在import阶段就会抛出AttributeError。
       修复方式：改用 typing.Any，避免依赖PyTorch内部私有类名。
    2. 实现AMP混合精度训练：
       原配置文件中 mixed_precision: true 从未被实际使用，现实现自动混合精度
       （仅在CUDA可用时生效，CPU环境下自动回退为全精度，保证跨设备兼容）。
    3. fit() 方法增加 try/finally：
       保证即使训练过程中抛出异常（如数据问题），TensorBoard的SummaryWriter
       依然会被正确flush和close，避免日志数据丢失。
    4. GPU数据传输增加 non_blocking=True（仅在pin_memory生效时有意义）。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mambabird.frontend.base_frontend import BaseFrontend
from mambabird.backbone.base_backbone import BaseBackbone
from mambabird.metrics.classification_metrics import compute_classification_metrics
from mambabird.trainer.tb_logger import TensorBoardLogger
from mambabird.utils.logger import get_logger

logger = get_logger(__name__)


class BirdSoundClassifier(nn.Module):
    """
    鸟声分类整体模型：声学前端 + 骨干网络 的组合封装。
    该类本身不包含任何训练逻辑，仅负责前向计算的组装，
    便于在模型保存/加载、ONNX导出等场景下作为单一nn.Module整体处理。
    """

    def __init__(self, frontend: BaseFrontend, backbone: BaseBackbone) -> None:
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.frontend(waveform)
        logits = self.backbone(features)
        return logits


class BaseTrainer:
    """
    标准训练器：实现单一损失（分类损失）下的训练循环与验证评估。
    """

    def __init__(
        self,
        model: BirdSoundClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_config: Dict[str, Any],
        experiment_dir: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        """
        参数：
            model: BirdSoundClassifier 实例（前端+骨干网络组合）。
            train_loader: 训练集 DataLoader。
            val_loader: 验证集 DataLoader。
            train_config: 来自实验YAML中 "train" 字段的配置字典。
            experiment_dir: 当前实验的产物根目录
                            （即 experiments/<exp_name>_<timestamp>/）。
            device: 计算设备字符串（"cuda" 或 "cpu"），若为None则自动检测。
        """
        self.train_config = train_config
        self.experiment_dir = Path(experiment_dir)

        # 自动设备检测：若配置要求cuda但当前环境不可用，则自动回退为cpu
        requested_device = device or train_config.get("device", "cpu")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("配置中指定使用cuda，但当前环境未检测到可用GPU，自动回退为cpu。")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(requested_device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # ---------------- AMP混合精度配置 ----------------
        # 仅在CUDA可用且配置开启mixed_precision时启用，CPU上AMP收益有限且兼容性较差，
        # 因此即使配置开启，CPU环境下也会自动禁用，保证训练不因设备差异而报错。
        self.use_amp = bool(self.train_config.get("mixed_precision", False)) and self.device.type == "cuda"
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("已启用AMP混合精度训练。")

        # TensorBoard日志记录器
        tb_log_dir = self.experiment_dir / "tb_logs"
        self.tb_logger = TensorBoardLogger(log_dir=tb_log_dir)

        # checkpoint保存目录
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_val_accuracy = 0.0

        # pin_memory是否生效（用于决定数据搬运时是否使用non_blocking=True）
        self._non_blocking = self.device.type == "cuda"

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """根据配置构建优化器（V1仅支持AdamW，未来可扩展注册器模式）。"""
        opt_cfg = self.train_config["optimizer"]
        optimizer_type = opt_cfg.get("type", "adamw").lower()

        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg["learning_rate"],
                weight_decay=opt_cfg.get("weight_decay", 0.0),
            )
        raise ValueError(f"未支持的优化器类型: {optimizer_type}")

    def _build_scheduler(self) -> Optional[Any]:
        """
        根据配置构建学习率调度器。

        【修复说明】返回类型标注由私有API `torch.optim.lr_scheduler._LRScheduler`
        改为 `Optional[Any]`，避免不同PyTorch版本下私有类名变动导致模块import失败。
        """
        scheduler_cfg = self.train_config.get("scheduler", {})
        scheduler_type = scheduler_cfg.get("type", None)

        if scheduler_type is None:
            return None
        if scheduler_type == "cosine_annealing":
            total_epochs = self.train_config["epochs"]
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs
            )
        raise ValueError(f"未支持的学习率调度器类型: {scheduler_type}")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        执行一个完整epoch的训练。

        参数：
            epoch: 当前epoch编号（从0开始）。

        返回：
            该epoch的平均训练损失等指标字典。
        """
        self.model.train()

        log_interval = self.train_config.get("log_interval_steps", 10)
        grad_clip_norm = self.train_config.get("gradient_clip_norm", None)

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (waveform, label) in enumerate(self.train_loader):
            waveform = waveform.to(self.device, non_blocking=self._non_blocking)
            label = label.to(self.device, non_blocking=self._non_blocking)

            self.optimizer.zero_grad()

            # ---------------- AMP前向 + 反向 ----------------
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(waveform)
                loss = self.criterion(logits, label)

            self.grad_scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                # 梯度裁剪前需先unscale，否则裁剪阈值会被AMP的scale系数影响
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if (batch_idx + 1) % log_interval == 0:
                self.tb_logger.log_scalar("train/loss_step", loss.item(), self.global_step)
                logger.info(
                    f"[Epoch {epoch}] [Step {batch_idx + 1}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f}"
                )

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        self.tb_logger.log_scalar("train/loss_epoch", avg_loss, epoch)

        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        在验证集上评估当前模型，计算损失、准确率与F1分数。

        参数：
            epoch: 当前epoch编号（用于TensorBoard记录）。

        返回：
            包含 val_loss / accuracy / f1_macro 的指标字典。
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []

        for waveform, label in self.val_loader:
            waveform = waveform.to(self.device, non_blocking=self._non_blocking)
            label = label.to(self.device, non_blocking=self._non_blocking)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(waveform)
                loss = self.criterion(logits, label)

            total_loss += loss.item()
            num_batches += 1

            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(label.cpu().tolist())

        avg_loss = total_loss / max(num_batches, 1)
        metrics = compute_classification_metrics(all_labels, all_predictions)
        metrics["val_loss"] = avg_loss

        self.tb_logger.log_scalars(
            {
                "val/loss_epoch": avg_loss,
                "val/accuracy": metrics["accuracy"],
                "val/f1_macro": metrics["f1_macro"],
            },
            step=epoch,
        )

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        保存模型checkpoint。

        参数：
            epoch: 当前epoch编号。
            is_best: 是否为当前验证集表现最优的模型（若是，额外保存为 best.pt）。
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
        }

        # 按epoch保存
        epoch_ckpt_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_ckpt_path)
        logger.info(f"已保存checkpoint: {epoch_ckpt_path}")

        # 保存最优模型
        if is_best:
            best_ckpt_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_ckpt_path)
            logger.info(f"已更新最优模型: {best_ckpt_path}")

    def fit(self) -> None:
        """
        执行完整训练流程：多个epoch的训练 + 验证 + 日志记录 + checkpoint保存。

        【修复说明】使用 try/finally 包裹整个训练循环，保证无论训练过程中是否
        抛出异常，TensorBoard的SummaryWriter都会被正确flush并close，避免日志丢失。
        """
        total_epochs = self.train_config["epochs"]
        save_interval = self.train_config.get("checkpoint_save_interval_epochs", 5)

        try:
            for epoch in range(total_epochs):
                train_metrics = self.train_one_epoch(epoch)
                val_metrics = self.validate(epoch)

                logger.info(
                    f"[Epoch {epoch}] train_loss={train_metrics['train_loss']:.4f} "
                    f"val_loss={val_metrics['val_loss']:.4f} "
                    f"val_accuracy={val_metrics['accuracy']:.4f} "
                    f"val_f1_macro={val_metrics['f1_macro']:.4f}"
                )

                is_best = val_metrics["accuracy"] > self.best_val_accuracy
                if is_best:
                    self.best_val_accuracy = val_metrics["accuracy"]

                if (epoch + 1) % save_interval == 0 or is_best or (epoch == total_epochs - 1):
                    self.save_checkpoint(epoch, is_best=is_best)

            logger.info(f"训练完成。最优验证集准确率: {self.best_val_accuracy:.4f}")
        finally:
            # 无论训练是否正常结束，都确保TensorBoard日志被正确flush和关闭
            self.tb_logger.close()