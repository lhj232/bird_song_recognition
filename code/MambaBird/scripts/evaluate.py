# -*- coding: utf-8 -*-
"""
scripts/evaluate.py

模型评估入口脚本。

【本次修复】
    原代码已生成 confusion_matrix_to_tensorboard_image()，但从未调用，
    导致混淆矩阵只保存了本地图片，未写入TensorBoard，与"所有评估必须支持TensorBoard"
    的项目要求不符。本次修复在保存本地图片的同时，将混淆矩阵图像一并写入TensorBoard。
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from mambabird.backbone.registry import build_backbone  # noqa: E402
from mambabird.data.dataloader_builder import build_dataloaders  # noqa: E402
from mambabird.evaluators.classification_evaluator import ClassificationEvaluator  # noqa: E402
from mambabird.evaluators.confusion_matrix_visualizer import (  # noqa: E402
    confusion_matrix_to_tensorboard_image,
    plot_confusion_matrix,
)
from mambabird.frontend.registry import build_frontend  # noqa: E402
from mambabird.trainer.base_trainer import BirdSoundClassifier  # noqa: E402
from mambabird.trainer.tb_logger import TensorBoardLogger  # noqa: E402
from mambabird.utils.config_parser import load_yaml_config  # noqa: E402
from mambabird.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="MambaBird 模型评估脚本")
    parser.add_argument("--config", type=str, required=True, help="实验配置YAML文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="待评估的checkpoint路径(.pt)")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="评估所使用的数据划分，默认val（验证集）",
    )
    return parser.parse_args()


@torch.no_grad()
def run_inference(model, dataloader, device) -> tuple:
    """
    在给定dataloader上执行推理，返回全部真实标签与预测标签列表。
    """
    model.eval()
    y_true_list = []
    y_pred_list = []

    for waveform, label in dataloader:
        waveform = waveform.to(device)
        logits = model(waveform)
        predictions = torch.argmax(logits, dim=1).cpu().tolist()

        y_pred_list.extend(predictions)
        y_true_list.extend(label.tolist())

    return y_true_list, y_pred_list


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.config)
    data_cfg = config["data"]
    frontend_cfg = config["frontend"]
    backbone_cfg = config["backbone"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    logger.info("正在加载数据集 ...")
    train_loader, val_loader, test_loader, label_to_idx, idx_to_label = build_dataloaders(data_cfg)
    num_classes = len(label_to_idx)
    label_names = [idx_to_label[i] for i in range(num_classes)]

    target_loader = val_loader if args.split == "val" else test_loader
    logger.info(f"评估数据划分: {args.split}，样本数: {len(target_loader.dataset)}")

    backbone_cfg["num_classes"] = num_classes
    frontend = build_frontend(frontend_cfg, sample_rate=data_cfg["sample_rate"])
    backbone = build_backbone(backbone_cfg)
    model = BirdSoundClassifier(frontend=frontend, backbone=backbone).to(device)

    checkpoint_path = Path(args.checkpoint)
    logger.info(f"正在加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("正在执行推理 ...")
    y_true, y_pred = run_inference(model, target_loader, device)

    evaluator = ClassificationEvaluator(num_classes=num_classes)
    result = evaluator.evaluate(y_true=y_true, y_pred=y_pred, label_names=label_names)
    result.print_report()

    output_dir = checkpoint_path.parent.parent / f"eval_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_json_path = output_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_full_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"评估指标已保存: {metrics_json_path}")

    # ---------------- 混淆矩阵：本地图片 ----------------
    cm_image_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        confusion_matrix=result.confusion_matrix,
        label_names=result.label_names,
        normalize=True,
        figure_title=f"Confusion Matrix ({args.split})",
        save_path=cm_image_path,
    )
    logger.info(f"混淆矩阵图像已保存: {cm_image_path}")

    # ---------------- TensorBoard记录 ----------------
    tb_logger = TensorBoardLogger(log_dir=output_dir / "tb_logs")
    tb_logger.log_scalars(result.to_summary_dict(), step=0)

    # 【修复】将混淆矩阵图像同步写入TensorBoard，而非只保存本地图片
    cm_image_chw = confusion_matrix_to_tensorboard_image(
        confusion_matrix=result.confusion_matrix,
        label_names=result.label_names,
        normalize=True,
        figure_title=f"Confusion Matrix ({args.split})",
    )
    tb_logger.log_image(f"confusion_matrix/{args.split}", cm_image_chw, step=0)

    tb_logger.close()
    logger.info(f"评估指标与混淆矩阵已写入TensorBoard: {output_dir / 'tb_logs'}")


if __name__ == "__main__":
    main()