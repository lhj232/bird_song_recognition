# MambaBird

## 项目简介

**MambaBird** 是大学生创新创业训练计划项目
《基于流形约束和拓扑收缩的复杂环境下多模态鸟声识别方法研究》的工程实现代码库。

本项目旨在解决复杂野外环境下鸟声识别面临的三大挑战：
1. 环境噪声干扰严重；
2. 鸟声特征高度多样化；
3. 野外部署对模型轻量化的迫切需求。

核心技术路线：

```
BirdCLEF2024 → LogMel / mCAF 前端 → Mamba 主干网络
            → 拓扑收缩损失（Topology Loss） → 知识蒸馏 → 量化 → FastAPI 部署
```

## 核心创新点

1. **mCAF（流形约束可学习声学滤波器）**：
   引入 Sinkhorn-Knopp 迭代算法，将滤波器权重矩阵约束在双随机矩阵流形上，
   实现能量守恒的自适应特征提取，消除自由学习导致的梯度爆炸问题。

2. **动态拓扑收缩机制**：
   基于持续同调计算贝蒂数，构建拓扑惩罚项，结合强化学习元控制器实现
   "训练早期抹平拓扑空洞、后期动态释放几何复杂度"的课程学习策略。

3. **多模型系统对比**：
   在统一数据集与统一拓扑模块下，系统对比 CNN 系（ResNet/EfficientNet/
   MobileNet）、Transformer 系（ViT/AST）与 Mamba 的识别性能。

4. **轻量化联合优化**：
   采用"先蒸馏、后剪枝、再量化"的联合优化流水线，并将模型转换为
   ONNX/TensorRT/TFLite 格式，部署到树莓派/Jetson Nano 等边缘设备。

## 目录结构概览

```
MambaBird/
├── configs/            # 全部 YAML 配置文件（数据/前端/骨干网络/拓扑/训练/蒸馏/量化/实验）
├── mambabird/           # 核心 Python 包（数据/前端/骨干网络/拓扑/损失/训练器/压缩/部署/评估/工具）
├── scripts/             # 可执行入口脚本（训练/评估/蒸馏/剪枝/量化/导出/边缘基准测试）
├── experiments/         # 自动生成的实验产物目录（配置快照/checkpoint/TensorBoard日志）
├── data_raw/             # 原始数据集（不纳入版本控制）
├── data_processed/        # 预处理后的特征缓存（不纳入版本控制）
├── tests/                # 单元测试（与 mambabird 包结构镜像）
├── docs/                 # 项目文档、消融实验记录与论文写作素材
├── deploy_service/        # 独立 FastAPI 部署子项目
└── notebooks/             # 探索性数据分析 Notebook
```

详细架构设计请参见 `docs/PRD.md` 与 `docs/architecture.md`。

## 环境安装

```bash
# 使用 conda 创建环境（推荐，保证可复现）
conda env create -f environment.yaml
conda activate mambabird

# 或使用 pip 安装依赖
pip install -r requirements.txt
```

## 快速开始

```bash
# 1. 准备数据：将 BirdCLEF2024 数据集放置于 data_raw/birdclef2024/ 目录下

# 2. 运行最小闭环训练（LogMel + ResNet 基线）
python scripts/train.py --config configs/experiments/exp01_mcaf_vs_logmel.yaml

# 3. 启动 TensorBoard 查看训练曲线
tensorboard --logdir experiments/

# 4. 启动 FastAPI 部署服务
cd deploy_service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 实验管理

所有论文实验均通过 `configs/experiments/` 下的单一 YAML 文件驱动，
运行后会在 `experiments/<exp_name>_<timestamp>/` 下自动生成：

- `config_snapshot.yaml`：实验当时的完整配置快照
- `git_commit.txt`：代码版本哈希
- `checkpoints/`：模型权重
- `tb_logs/`：TensorBoard 日志
- `metrics.json`：最终指标汇总

以保证每个实验结果均可精确复现。

## 项目信息

- 项目类别：国家级大学生创新创业训练计划
- 所属学院：北京林业大学信息学院
- 指导教师：王新阳

## 开发规范

1. 所有代码遵循工业级 Python 规范，使用 PyTorch 实现。
2. 所有模块严格解耦，禁止将多个功能写入同一文件。
3. 所有配置统一采用 YAML 管理，禁止在代码中硬编码超参数。
4. 所有训练过程必须支持 TensorBoard 可视化。
5. 所有实验必须保证可复现（固定随机种子、记录配置快照与代码版本）。
