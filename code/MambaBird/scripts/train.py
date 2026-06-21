# -*- coding: utf-8 -*-
"""
scripts/train.py

V1 训练入口脚本：BirdCLEF2024 + LogMel + ResNet18。

使用方法：
    python scripts/train.py --config configs/experiments/v1_logmel_resnet18.yaml

职责：
    本脚本只负责"解析配置 -> 构建数据/前端/骨干网络/训练器 -> 启动训练"，
    不包含任何具体业务逻辑，所有逻辑均委托给 mambabird 包内的对应模块，
    保证训练入口保持轻量、可读，且核心逻辑可被单元测试覆盖。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 保证脚本可以在项目根目录之外被直接调用时，仍能正确import mambabird包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mambabird.backbone.registry import build_backbone  # noqa: E402
from mambabird.data.dataloader_builder import build_dataloaders  # noqa: E402
from mambabird.frontend.registry import build_frontend  # noqa: E402
from mambabird.trainer.base_trainer import BaseTrainer, BirdSoundClassifier  # noqa: E402
from mambabird.trainer.reproducibility import save_git_commit_hash, set_seed  # noqa: E402
from mambabird.utils.config_parser import load_yaml_config, save_yaml_config  # noqa: E402
from mambabird.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="MambaBird V1 训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="实验配置YAML文件路径，例如 configs/experiments/v1_logmel_resnet18.yaml",
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


def main() -> None:
    args = parse_args()

    # 1. 加载配置
    config = load_yaml_config(args.config)

    exp_cfg = config["experiment"]
    data_cfg = config["data"]
    frontend_cfg = config["frontend"]
    backbone_cfg = config["backbone"]
    train_cfg = config["train"]

    # 2. 固定随机种子，保证可复现
    random_seed = exp_cfg.get("random_seed", 42)
    set_seed(random_seed, deterministic=True)
    logger.info(f"已固定随机种子: {random_seed}")

    # 3. 创建实验产物目录并保存配置快照
    experiment_dir = prepare_experiment_dir(config)

    # 4. 构建DataLoader与标签映射
    logger.info("正在构建数据集与DataLoader ...")
    train_loader, val_loader, test_loader, label_to_idx, idx_to_label = build_dataloaders(data_cfg)
    num_classes = len(label_to_idx)
    logger.info(f"数据集解析完成，共 {num_classes} 个类别。")

    # 将动态解析出的类别数注入骨干网络配置（覆盖配置文件中的占位值-1）
    backbone_cfg["num_classes"] = num_classes

    # 5. 构建声学前端
    logger.info(f"正在构建声学前端: {frontend_cfg['type']} ...")
    frontend = build_frontend(frontend_cfg, sample_rate=data_cfg["sample_rate"])

    # 6. 构建骨干网络
    logger.info(f"正在构建骨干网络: {backbone_cfg['type']} ...")
    backbone = build_backbone(backbone_cfg)

    # 7. 组装完整模型
    model = BirdSoundClassifier(frontend=frontend, backbone=backbone)

    # 8. 构建训练器并启动训练
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=train_cfg,
        experiment_dir=experiment_dir,
    )

    logger.info("开始训练 ...")
    trainer.fit()

    logger.info(f"V1训练流程全部完成，实验产物位于: {experiment_dir}")


if __name__ == "__main__":
    main()