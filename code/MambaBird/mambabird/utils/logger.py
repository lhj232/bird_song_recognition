# -*- coding: utf-8 -*-
"""
mambabird/utils/logger.py

统一日志工具模块。
负责创建项目内统一格式的 Python logging 实例，避免各模块各自配置日志格式
导致输出风格不一致，便于训练过程中在终端与日志文件中追踪问题。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    获取（或创建）一个统一格式的 logger 实例。

    参数：
        name: logger 名称，通常传入 __name__，便于区分日志来源模块。
        log_file: 若指定，则同时将日志写入该文件路径（自动创建父目录）。

    返回：
        配置完成的 logging.Logger 实例。
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler（多次调用 get_logger 时不会产生重复日志输出）
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 终端输出 handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出 handler（可选）
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger