"""
Set Transformer 编码器
对应论文第 "Encoder Model" 节：
  "We implement q_phi(·) following the set transformer encoder approach (Lee et al. 2019)
   to ensure permutation invariance with respect to the number and order of active anchors."

原始论文：Lee et al. 2019, "Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks", ICML 2019.

核心思想：
  - 锚点观测集合 o(α_t) = {(d_t^k, p^k) | α_t^k=1} 是一个无序集合
  - 普通 MLP/RNN 对输入顺序敏感，而 Set Transformer 通过自注意力机制
    保证对集合元素的排列不变性（Permutation Invariance）
  - SAB (Set Attention Block): 集合内的元素互相做自注意力，提取元素间关系
  - PMA (Pooling by Multihead Attention): 将可变大小的集合池化为固定维度向量
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """
    标准多头注意力（Multi-head Attention）

    原理：将 Q、K、V 分为 h 个头，每个头在低维子空间上做点积注意力，
    最后拼接所有头的输出并线性投影。

    Args:
        d_model: 输入特征维度
        num_heads: 注意力头数
        dropout: dropout 概率
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q、K、V 的线性投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            Q: [B, N_q, d_model]
            K: [B, N_k, d_model]
            V: [B, N_k, d_model]
            mask: 可选的注意力掩码

        Returns:
            output: [B, N_q, d_model]
        """
        B = Q.size(0)

        # 线性投影并拆分成多头：[B, N, d_model] → [B, num_heads, N, d_k]
        Q = self.W_q(Q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力：scores = QK^T / sqrt(d_k)，shape [B, h, N_q, N_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax 归一化并 dropout
        attn = self.dropout(F.softmax(scores, dim=-1))

        # 加权求和：[B, h, N_q, d_k]
        out = torch.matmul(attn, V)

        # 拼接多头结果并输出投影：[B, N_q, d_model]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


class SAB(nn.Module):
    """
    Set Attention Block（集合注意力块）

    对集合内的所有元素做自注意力（Q=K=V=X），捕获元素间的成对关系。
    保证排列不变性：改变输入顺序，输出对应元素不变。

    结构：MultiheadAttention → LayerNorm → FFN → LayerNorm

    Args:
        d_model: 特征维度
        num_heads: 注意力头数
        ffn_dim: 前馈网络隐藏维度（通常 4 * d_model）
        dropout: dropout 概率
    """

    def __init__(self, d_model: int, num_heads: int,
                 ffn_dim: int = None, dropout: float = 0.0):
        super().__init__()
        ffn_dim = ffn_dim or (4 * d_model)

        self.attn = MultiheadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [B, N, d_model]  —  N 个集合元素

        Returns:
            [B, N, d_model]  —  经注意力更新后的集合表示
        """
        # 自注意力 + 残差连接 + LayerNorm
        X = self.norm1(X + self.dropout(self.attn(X, X, X)))
        # 前馈网络 + 残差连接 + LayerNorm
        X = self.norm2(X + self.dropout(self.ffn(X)))
        return X


class PMA(nn.Module):
    """
    Pooling by Multihead Attention（基于多头注意力的池化）

    用 k 个可学习的"种子向量"（Seed Vectors）作为 Query，
    对集合元素做交叉注意力，将可变大小的集合池化为固定大小（k 个向量）。
    当 k=1 时输出单个向量作为集合的聚合表示。

    Args:
        d_model: 特征维度
        num_heads: 注意力头数
        num_seeds: 种子向量数量（默认 1，输出一个聚合向量）
    """

    def __init__(self, d_model: int, num_heads: int, num_seeds: int = 1):
        super().__init__()
        # 可学习的种子向量作为 Query
        self.S = nn.Parameter(torch.randn(1, num_seeds, d_model))
        self.attn = MultiheadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: [B, N, d_model]  —  集合元素

        Returns:
            [B, num_seeds, d_model]  —  池化后的固定大小表示
        """
        B = Z.size(0)
        # 将种子向量扩展到 batch 维度
        S = self.S.expand(B, -1, -1)  # [B, num_seeds, d_model]
        # 用种子向量查询集合元素
        H = self.norm(S + self.attn(S, Z, Z))
        H = self.norm2(H + self.ffn(H))
        return H


class SetTransformerEncoder(nn.Module):
    """
    Set Transformer 编码器，用于对锚点观测集合编码。

    输入：锚点观测集合 o(α_t) = {(d_t^k, p^k) | α_t^k=1}
          每个元素是 (distance, anchor_x, anchor_y)，维度为 3
    输出：集合的聚合表示向量，维度为 d_model

    结构：
      1. 输入投影：3 → d_model（将每个锚点观测映射到 d_model 维）
      2. SAB × num_sab_layers：自注意力提取元素间关系
      3. PMA：将集合池化为单个向量

    Args:
        input_dim: 每个集合元素的维度，默认 3 = (1个距离 + 2D位置)
        d_model: 内部特征维度
        num_heads: 注意力头数
        num_sab_layers: SAB 层数
    """

    def __init__(self, input_dim: int = 3, d_model: int = 64,
                 num_heads: int = 4, num_sab_layers: int = 2):
        super().__init__()

        # 将原始观测 (d, px, py) 投影到 d_model 维
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
        )

        # 堆叠多层 SAB，捕获锚点间的相互关系
        self.sab_layers = nn.ModuleList(
            [SAB(d_model, num_heads) for _ in range(num_sab_layers)]
        )

        # PMA 将可变数量的锚点观测池化为单个向量
        self.pma = PMA(d_model, num_heads, num_seeds=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, K, 3]  — B 个样本，每个样本有 K 个活跃锚点的观测
                              obs[b, k] = [d_t^k, p_x^k, p_y^k]

        Returns:
            [B, d_model]  —  整个锚点集合的聚合表示
        """
        # 输入投影：[B, K, 3] → [B, K, d_model]
        X = self.input_proj(obs)

        # 通过多层 SAB：[B, K, d_model]
        for sab in self.sab_layers:
            X = sab(X)

        # PMA 池化：[B, K, d_model] → [B, 1, d_model] → [B, d_model]
        out = self.pma(X)          # [B, 1, d_model]
        return out.squeeze(1)      # [B, d_model]
