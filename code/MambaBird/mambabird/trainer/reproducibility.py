# -*- coding: utf-8 -*-
"""
mambabird/trainer/reproducibility.py

可复现性工具模块。

【本次修复】
    get_git_commit_hash() 原先未指定 subprocess 的 cwd 参数，
    若脚本运行时的当前工作目录不是项目根目录（Windows用户通过IDE/双击运行时常见），
    会获取到错误的commit哈希，甚至在该目录不属于任何git仓库时直接报错。
    修复方式：显式将 cwd 设置为项目根目录（基于本文件路径反推），保证无论从何处启动
    脚本，都能获取到正确的代码版本哈希。
"""

import random
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

# 项目根目录：mambabird/trainer/reproducibility.py 向上回退两级即为项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    固定 Python / NumPy / PyTorch 的全局随机种子。

    参数：
        seed: 随机种子数值。
        deterministic: 是否开启 CUDA 确定性计算模式
                        （开启后结果完全可复现，但可能略微降低训练速度）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_git_commit_hash() -> str:
    """
    获取当前代码仓库的 git commit 哈希值，用于记录实验对应的代码版本。
    若当前目录不是git仓库或git不可用，则返回"unknown"，不影响训练主流程。

    【修复】显式指定 cwd=_PROJECT_ROOT，避免脚本启动目录不一致（尤其Windows环境下
    通过IDE/双击方式运行时工作目录可能不是项目根目录）导致获取错误或报错。

    返回：
        git commit 哈希字符串（或 "unknown"）。
    """
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(_PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        return commit_hash
    except Exception:
        return "unknown"


def save_git_commit_hash(save_path: Union[str, Path]) -> None:
    """
    将当前git commit哈希写入实验目录下的文件，便于复现实验时核对代码版本。

    参数：
        save_path: 保存路径（通常为 experiments/<exp_name>/git_commit.txt）。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    commit_hash = get_git_commit_hash()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(commit_hash + "\n")