# -*- coding: utf-8 -*-
"""
mambabird/backbone/mamba/selective_ssm.py

选择性状态空间核心算子模块（Selective State Space Model, SSM）。

本模块实现 Mamba 论文中"选择性扫描"（Selective Scan）机制的纯 PyTorch 版本：
    - 不依赖 mamba-ssm / causal-conv1d 等需要CUDA编译的第三方库，
      保证在CPU、Windows、无CUDA编译环境下也能正确运行（工程可移植性优先）。
    - 数据依赖的离散化参数（delta / B / C）使得模型能够根据输入内容
      动态决定"记住"或"遗忘"历史信息，是Mamba区别于传统CNN/RNN的核心机制。

设计说明：
    本实现采用顺序扫描（sequential scan），在时间维度上逐步递推隐状态，
    正确性优先于极致效率；当前版本足以支撑V2阶段"多模型对比实验"中
    Mamba与CNN/Transformer的公平对比，后续可在确认正确性后再替换为
    并行化的chunked scan或CUDA kernel以提升训练速度。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    因果卷积模块：保证第t个时间步的输出只依赖于 <= t 的输入，
    避免Mamba的局部卷积模块"偷看"未来信息（这对流式/边缘部署场景尤为重要）。
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,          # depthwise卷积，逐通道独立处理，符合Mamba原始设计
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: 形状为 (batch_size, channels, seq_len) 的输入张量。
        返回：
            形状与输入相同的因果卷积输出。
        """
        out = self.conv(x)
        # 裁剪掉右侧多余的padding，保证输出长度与输入一致且满足因果性
        return out[:, :, : -self.padding] if self.padding > 0 else out


class SelectiveSSMBlock(nn.Module):
    """
    单个Mamba选择性状态空间模块（对应论文中的Mamba Block）。

    数据流：
        输入 x (B, T, d_model)
        -> 线性升维并拆分为 (主分支, 门控分支)，各为 d_inner
        -> 主分支经因果卷积 + SiLU激活
        -> 数据依赖地生成 delta(离散化步长) / B / C 参数
        -> 选择性扫描，得到序列隐状态输出
        -> 与门控分支(SiLU激活)逐元素相乘
        -> 线性降维回 d_model，与输入残差相加
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ) -> None:
        """
        参数：
            d_model: 模块输入/输出特征维度。
            d_state: SSM隐状态维度（每个通道独立维护一个d_state维的隐状态向量）。
            d_conv: 因果卷积核大小。
            expand_factor: 内部升维倍数，d_inner = d_model * expand_factor。
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor

        # 输入投影：同时生成主分支与门控分支，故输出维度为 2 * d_inner
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        # 因果深度卷积，作用于主分支
        self.causal_conv = CausalConv1d(channels=self.d_inner, kernel_size=d_conv)

        # 数据依赖参数生成：delta（标量/通道）、B、C（各为d_state维）
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + 1)

        # delta 的可学习偏置投影（对应Mamba中的dt_proj，用于稳定初始化）
        self.delta_proj = nn.Linear(1, self.d_inner, bias=True)

        # 状态转移矩阵A的对数参数化（保证离散化后A为负，系统稳定收敛）
        # A_log形状: (d_inner, d_state)，每个通道独立的状态空间参数
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))

        # 跳跃连接系数D（对应原始输入的直通项）
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影：d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        B_param: torch.Tensor,
        C_param: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行选择性扫描（核心递推计算）。

        递推公式（每个通道独立）：
            h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t
            y_t = C_t · h_t

        参数：
            x: 形状为 (batch, seq_len, d_inner) 的主分支输入。
            delta: 形状为 (batch, seq_len, d_inner) 的数据依赖离散化步长。
            B_param: 形状为 (batch, seq_len, d_state) 的数据依赖输入矩阵。
            C_param: 形状为 (batch, seq_len, d_state) 的数据依赖输出矩阵。

        返回：
            形状为 (batch, seq_len, d_inner) 的扫描输出。
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = self.d_state

        # A: (d_inner, d_state)，恒为负，保证系统稳定
        A = -torch.exp(self.A_log.to(x.dtype))  # (d_inner, d_state)

        # 离散化状态转移系数: exp(delta_t * A)，形状 (batch, seq_len, d_inner, d_state)
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # 输入项: delta_t * B_t * x_t，形状 (batch, seq_len, d_inner, d_state)
        delta_B_x = (
            delta.unsqueeze(-1)
            * B_param.unsqueeze(2)   # (batch, seq_len, 1, d_state) -> broadcast到d_inner
            * x.unsqueeze(-1)        # (batch, seq_len, d_inner, 1)
        )

        # 顺序递推隐状态（沿seq_len维度循环），h形状: (batch, d_inner, d_state)
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_x[:, t]
            # y_t = C_t · h_t，对d_state维求和
            y_t = torch.einsum("bds,bs->bd", h, C_param[:, t])
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: 形状为 (batch_size, seq_len, d_model) 的输入序列特征。

        返回：
            形状为 (batch_size, seq_len, d_model) 的输出序列特征（已与输入残差相加）。
        """
        residual = x

        # 1. 输入投影并拆分为主分支与门控分支
        x_and_z = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_main, z_gate = x_and_z.chunk(2, dim=-1)  # 各 (batch, seq_len, d_inner)

        # 2. 因果卷积 + SiLU激活（需先转换为 (batch, d_inner, seq_len) 以适配Conv1d）
        x_main = x_main.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_main = self.causal_conv(x_main)
        x_main = F.silu(x_main)
        x_main = x_main.transpose(1, 2)  # (batch, seq_len, d_inner)

        # 3. 生成数据依赖参数：delta / B / C
        ssm_params = self.x_proj(x_main)  # (batch, seq_len, 2*d_state + 1)
        delta_raw, B_param, C_param = torch.split(
            ssm_params, [1, self.d_state, self.d_state], dim=-1
        )

        # delta需为正数（离散化步长），通过softplus保证正值，并经delta_proj扩展到d_inner维度
        delta = F.softplus(self.delta_proj(delta_raw))  # (batch, seq_len, d_inner)

        # 4. 执行选择性扫描
        y = self._selective_scan(x_main, delta, B_param, C_param)

        # 5. 跳跃连接（直通项）+ 门控
        y = y + x_main * self.D
        y = y * F.silu(z_gate)

        # 6. 输出投影回 d_model，并与原始输入残差相加
        out = self.out_proj(y)
        return out + residual