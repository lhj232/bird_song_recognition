# -*- coding: utf-8 -*-
"""
scripts/train_resnet.py

ResNet18 专用训练入口脚本：BirdCLEF2024 + LogMel + ResNet18。

功能：
    1. 自动读取实验YAML配置（数据/前端/骨干网络/训练参数）。
    2. 自动加载BirdCLEF2024数据集并完成训练/验证/测试划分。
    3. 自动执行ResNet18模型训练（含TensorBoard记录）。
    4. 自动保存验证集表现最优的模型（best.pt）。
    5. 训练结束后自动在验证集上输出最终的 Accuracy 与 F1（宏平均）指标。

使用方法：
    python scripts/train_resnet.py --config configs/experiments/v1_logmel_resnet18.yaml

设计说明：
    本脚本是 scripts/train.py 的"ResNet专用"版本，逻辑上保持与通用训练入口一致，
    但额外增加了：
        - 强制校验骨干网络类型必须为 resnet 系列，避免误用配置文件；
        - 训练结束后单独执行一次"最优模型"加载与最终评估，
          并将最终 Accuracy / F1 以醒目格式打印到终端，
          便于答辩/汇报时直接截图展示。
    所有底层逻辑均委托给 mambabird 包内对应模块，本脚本不包含任何业务逻辑实现。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 保证脚本可以在项目根目录之外被直接调用时，仍能正确import mambabird包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from mambabird.backbone.registry import build_backbone  # noqa: E402
from mambabird.data.dataloader_builder import build_dataloaders  # noqa: E402
from mambabird.frontend.registry import build_frontend  # noqa: E402
from mambabird.metrics.classification_metrics import compute_classification_metrics  # noqa: E402
from mambabird.trainer.base_trainer import BaseTrainer, BirdSoundClassifier  # noqa: E402
from mambabird.trainer.reproducibility import save_git_commit_hash, set_seed  # noqa: E402
from mambabird.utils.config_parser import load_yaml_config, save_yaml_config  # noqa: E402
from mambabird.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# 本脚本只允许使用的骨干网络类型（ResNet系列），避免配置误用
ALLOWED_BACKBONE_TYPES = {"resnet18", "resnet34", "resnet50"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="MambaBird ResNet18 专用训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/v1_logmel_resnet18.yaml",
        help="实验配置YAML文件路径（默认: configs/experiments/v1_logmel_resnet18.yaml）",
    )
    return parser.parse_args()


def prepare_experiment_dir(config: dict) -> Path:
    """
    创建本次实验的产物目录（带时间戳），并保存配置快照与git commit哈希。

    参数：
        config: 完整实验配置字典。

    返回：
        本次实验产物目录的绝对路径。
    """
    exp_cfg = config["experiment"]
    exp_name = exp_cfg["name"]
    output_root = Path(exp_cfg.get("output_root", "experiments"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_root / f"{exp_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置快照，保证实验可复现
    save_yaml_config(config, experiment_dir / "config_snapshot.yaml")

    # 保存git commit哈希，记录代码版本
    save_git_commit_hash(experiment_dir / "git_commit.txt")

    logger.info(f"实验产物目录已创建: {experiment_dir}")
    return experiment_dir


@torch.no_grad()
def evaluate_with_best_checkpoint(
    model: BirdSoundClassifier,
    checkpoint_path: Path,
    val_loader,
    device: torch.device,
) -> dict:
    """
    加载最优checkpoint，在验证集上重新评估一次，得到最终的Accuracy/F1指标。

    参数：
        model: 已构建好结构的 BirdSoundClassifier 实例（结构需与训练时一致）。
        checkpoint_path: best.pt 的文件路径。
        val_loader: 验证集 DataLoader。
        device: 计算设备。

    返回：
        包含 "accuracy" 与 "f1_macro" 的指标字典。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    for waveform, label in val_loader:
        waveform = waveform.to(device)
        label = label.to(device)

        logits = model(waveform)
        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(label.cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_predictions)
    return metrics


def print_final_metrics(metrics: dict, experiment_dir: Path) -> None:
    """
    以醒目格式将最终指标打印到终端，便于答辩/汇报截图。

    参数：
        metrics: 包含 "accuracy" 与 "f1_macro" 的指标字典。
        experiment_dir: 实验产物目录（用于在输出中提示checkpoint位置）。
    """
    separator = "=" * 60
    print("\n" + separator)
    print(" MambaBird V1 (LogMel + ResNet18) 最终验证集评估结果")
    print(separator)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1(macro): {metrics['f1_macro']:.4f}")
    print(f"  最优模型路径: {experiment_dir / 'checkpoints' / 'best.pt'}")
    print(separator + "\n")


def main() -> None:
    args = parse_args()

    # 1. 加载配置
    config = load_yaml_config(args.config)

    exp_cfg = config["experiment"]
    data_cfg = config["data"]
    frontend_cfg = config["frontend"]
    backbone_cfg = config["backbone"]
    train_cfg = config["train"]

    # 2. 校验骨干网络类型，确保本脚本只用于ResNet系列训练
    backbone_type = backbone_cfg.get("type", "")
    if backbone_type not in ALLOWED_BACKBONE_TYPES:
        raise ValueError(
            f"train_resnet.py 仅支持ResNet系列骨干网络，"
            f"当前配置中的backbone.type='{backbone_type}' 不被允许。"
            f"允许的类型: {sorted(ALLOWED_BACKBONE_TYPES)}"
        )

    # 3. 固定随机种子，保证可复现
    random_seed = exp_cfg.get("random_seed", 42)
    set_seed(random_seed, deterministic=True)
    logger.info(f"已固定随机种子: {random_seed}")

    # 4. 创建实验产物目录并保存配置快照
    experiment_dir = prepare_experiment_dir(config)

    # 5. 自动加载BirdCLEF2024数据集，构建DataLoader与标签映射
    logger.info("正在加载 BirdCLEF2024 数据集 ...")
    train_loader, val_loader, test_loader, label_to_idx, idx_to_label = build_dataloaders(data_cfg)
    num_classes = len(label_to_idx)
    logger.info(f"数据集加载完成，共解析出 {num_classes} 个物种类别。")
    logger.info(
        f"训练集样本数: {len(train_loader.dataset)} | "
        f"验证集样本数: {len(val_loader.dataset)} | "
        f"测试集样本数: {len(test_loader.dataset)}"
    )

    # 将动态解析出的类别数注入骨干网络配置（覆盖配置文件中的占位值-1）
    backbone_cfg["num_classes"] = num_classes

    # 6. 构建LogMel声学前端
    logger.info(f"正在构建声学前端: {frontend_cfg['type']} ...")
    frontend = build_frontend(frontend_cfg, sample_rate=data_cfg["sample_rate"])

    # 7. 构建ResNet18骨干网络
    logger.info(f"正在构建骨干网络: {backbone_cfg['type']} ...")
    backbone = build_backbone(backbone_cfg)

    # 8. 组装完整模型（LogMel前端 + ResNet18骨干网络）
    model = BirdSoundClassifier(frontend=frontend, backbone=backbone)

    # 9. 构建训练器（内部已封装TensorBoard记录与checkpoint保存逻辑）
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=train_cfg,
        experiment_dir=experiment_dir,
    )

    # 10. 启动自动训练（每个epoch自动验证、自动保存best模型）
    logger.info("开始自动训练 ResNet18 ...")
    trainer.fit()
    logger.info(
        f"训练完成。训练过程中最优验证集准确率: {trainer.best_val_accuracy:.4f}"
    )

    # 11. 加载最优模型，在验证集上重新评估，输出最终 Accuracy / F1
    best_checkpoint_path = experiment_dir / "checkpoints" / "best.pt"
    if best_checkpoint_path.exists():
        logger.info("正在加载最优模型(best.pt)进行最终评估 ...")
        final_metrics = evaluate_with_best_checkpoint(
            model=model,
            checkpoint_path=best_checkpoint_path,
            val_loader=val_loader,
            device=trainer.device,
        )
        print_final_metrics(final_metrics, experiment_dir)
    else:
        logger.warning(
            "未找到 best.pt，可能训练epoch数过少导致未触发最优模型保存逻辑。"
        )

    logger.info(f"全部流程结束，实验产物位于: {experiment_dir}")


if __name__ == "__main__":
    print("=== train_resnet.py main() 已启动 ===") 
    main()