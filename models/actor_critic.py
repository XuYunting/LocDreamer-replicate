"""
基于强化学习的锚点调度器（Actor-Critic）

对应论文 "RL-Based Anchor Scheduling in LocDreamer" 节及公式 (14)-(17)。

核心思想：
  - Actor（策略网络）根据当前 DSSM 状态 s_t = (z_t^prior, h_t) 选择最优锚点子集
  - Critic（价值网络）估计未来累积奖励，用于计算优势函数降低策略梯度方差
  - 奖励 R_t = L_recon_t：选中锚点的测距重建对数概率（信息量越大奖励越高）
  - 使用 REINFORCE 算法（带基线）更新 Actor

动作空间处理：
  - 目标：从 A=5 个锚点中选 Kt=3 个（恰好满足资源约束 Σα_k=Kt）
  - 方案：枚举所有 C(A, Kt) = C(5,3) = 10 种组合，Actor 输出组合概率
  - 优点：自动满足约束，且对于 A≤8 的场景组合数可控（最多 C(8,4)=70）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from typing import Tuple, List, Optional


class Actor(nn.Module):
    """
    锚点调度策略网络（式 14）：
      α_t ~ π_{φ_α}(s_t), s_t = (z_t^prior, h_t)

    实现方式：枚举所有合法调度方案（C(A, Kt) 种组合），输出各方案的选择概率。
    这保证了硬约束 Σα_k^t = Kt 被精确满足。

    训练：使用 REINFORCE（策略梯度，式 16）：
      L_actor = -E[ log π(α_t | s_t) * Adv(s_t, α_t) ]
    其中 Adv(s_t, α_t) = Σ_{τ≥t} γ^{τ-t} R_τ - V_{φ_V}(s_t)

    Args:
        state_dim:   z_t^prior 的维度 Z=4
        hidden_dim:  h_t 的维度 H=50
        num_anchors: A — 候选锚点总数
        num_scheduled: Kt — 每步选取的锚点数
        mlp_hidden:  MLP 隐藏层维度
    """

    def __init__(self, state_dim: int, hidden_dim: int,
                 num_anchors: int, num_scheduled: int,
                 mlp_hidden: int = 64):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_scheduled = num_scheduled

        # 预计算并注册所有合法调度方案
        # combos[i] = (k1, k2, ..., kKt) 第 i 种选法的锚点索引
        combos = list(combinations(range(num_anchors), num_scheduled))
        self.num_combos = len(combos)   # C(A, Kt)，如 A=5, Kt=3 → 10 种

        # 将组合列表注册为 buffer（不参与梯度，但跟随模型移至 GPU）
        # shape: [num_combos, Kt]
        combo_tensor = torch.tensor(combos, dtype=torch.long)
        self.register_buffer('combos', combo_tensor)

        # 策略 MLP：输入 s_t = [z_t^prior, h_t]，输出 num_combos 维 logits
        # 输入维度 = Z + H = 4 + 50 = 54
        input_dim = state_dim + hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, self.num_combos),
        )

    def forward(self, z_prior: torch.Tensor,
                h_t: torch.Tensor) -> torch.Tensor:
        """
        计算所有调度方案的概率分布。

        Args:
            z_prior: [B, Z]  — 当前先验状态
            h_t:     [B, H]  — RNN 隐藏状态

        Returns:
            probs: [B, num_combos]  — 每种调度方案的选择概率（softmax）
        """
        s_t = torch.cat([z_prior, h_t], dim=-1)   # [B, Z+H]
        logits = self.mlp(s_t)                     # [B, num_combos]
        probs = F.softmax(logits, dim=-1)          # [B, num_combos]
        return probs

    def sample_action(self, z_prior: torch.Tensor,
                      h_t: torch.Tensor,
                      greedy: bool = False
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样调度动作（Algorithm 1, Line 17：α_t ~ π_{θ_α}(z_t^prior, h_t)）

        Args:
            z_prior: [B, Z]
            h_t:     [B, H]
            greedy:  True → argmax（评估模式），False → 随机采样（训练模式）

        Returns:
            selected_idx:  [B, Kt]       — 选中锚点的索引
            combo_idx:     [B]           — 所选方案在 combos 中的编号
            log_prob:      [B]           — log π(α_t | s_t)，用于策略梯度
        """
        probs = self.forward(z_prior, h_t)   # [B, num_combos]

        if greedy:
            # 贪婪选择：选概率最高的方案
            combo_idx = probs.argmax(dim=-1)  # [B]
        else:
            # 随机采样（按概率分布）
            dist = torch.distributions.Categorical(probs)
            combo_idx = dist.sample()         # [B]

        # 根据方案编号获取被选中的锚点索引
        # combos: [num_combos, Kt]，combo_idx: [B] → selected_idx: [B, Kt]
        selected_idx = self.combos[combo_idx]  # [B, Kt]

        # 计算 log π(α_t | s_t)，用于 REINFORCE 损失
        log_prob = torch.log(probs.gather(1, combo_idx.unsqueeze(1)).squeeze(1) + 1e-8)

        return selected_idx, combo_idx, log_prob

    def get_log_prob(self, z_prior: torch.Tensor, h_t: torch.Tensor,
                     combo_idx: torch.Tensor) -> torch.Tensor:
        """
        给定状态和已执行的动作，重新计算 log π（用于 REINFORCE 的离线更新）

        Args:
            z_prior:   [B, Z]
            h_t:       [B, H]
            combo_idx: [B]  — 已采样的方案编号

        Returns:
            log_prob: [B]
        """
        probs = self.forward(z_prior, h_t)   # [B, num_combos]
        selected_probs = probs.gather(1, combo_idx.unsqueeze(1)).squeeze(1)  # [B]
        return torch.log(selected_probs + 1e-8)

    def get_entropy(self, z_prior: torch.Tensor,
                    h_t: torch.Tensor) -> torch.Tensor:
        """
        计算策略熵（可用于熵正则化，鼓励探索）

        Returns:
            entropy: [B]
        """
        probs = self.forward(z_prior, h_t)
        dist = torch.distributions.Categorical(probs)
        return dist.entropy()   # [B]

    def get_alpha_mask(self, combo_idx: torch.Tensor) -> torch.Tensor:
        """
        将方案编号转换为二值掩码向量 α_t ∈ {0,1}^A（式 14 中的 α_t）

        Args:
            combo_idx: [B]  — 方案编号

        Returns:
            mask: [B, A]  — 二值调度向量，mask[b, k]=1 表示锚点 k 被激活
        """
        B = combo_idx.shape[0]
        mask = torch.zeros(B, self.num_anchors, device=combo_idx.device)
        selected = self.combos[combo_idx]   # [B, Kt]
        # 在对应位置填 1
        mask.scatter_(1, selected, 1.0)
        return mask


class Critic(nn.Module):
    """
    价值函数网络（式 15）：
      V_{φ_V}(s_t) = E[ Σ_{τ≥t} γ^{τ-t} R_τ ]

    Critic 估计从当前状态出发的期望折扣累积奖励，用于计算优势函数：
      Adv(s_t, α_t) = G_t - V_{φ_V}(s_t)
    其中 G_t = Σ_{τ≥t} γ^{τ-t} R_τ 是实际折扣回报。

    训练目标（式 17）：
      L_critic = E[ (V_{φ_V}(s_t) - G_t)² ]

    Args:
        state_dim:  z_t^prior 的维度 Z=4
        hidden_dim: h_t 的维度 H=50
        mlp_hidden: MLP 隐藏层维度
    """

    def __init__(self, state_dim: int, hidden_dim: int, mlp_hidden: int = 64):
        super().__init__()

        # 价值 MLP：输入 s_t = [z_t^prior, h_t]，输出标量 V(s_t)
        input_dim = state_dim + hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 1),   # 输出标量
        )

    def forward(self, z_prior: torch.Tensor,
                h_t: torch.Tensor) -> torch.Tensor:
        """
        估计状态价值。

        Args:
            z_prior: [B, Z]  — 先验状态
            h_t:     [B, H]  — RNN 隐藏状态

        Returns:
            value: [B, 1]  — 状态价值估计 V(s_t)
        """
        s_t = torch.cat([z_prior, h_t], dim=-1)  # [B, Z+H]
        return self.mlp(s_t)                       # [B, 1]


def compute_returns(rewards: List[float], gamma: float = 0.99,
                    device: torch.device = None) -> torch.Tensor:
    """
    计算折扣累积回报 G_t = Σ_{τ≥t} γ^{τ-t} R_τ（对应式 15 的期望目标）

    从后往前递推：G_T = R_T, G_t = R_t + γ * G_{t+1}

    Args:
        rewards: 长度 T 的奖励列表 [R_0, R_1, ..., R_{T-1}]
        gamma:   折扣因子 γ（Table 1: 0.99）
        device:  目标设备

    Returns:
        returns: [T]  — 每步的折扣回报
    """
    T = len(rewards)
    returns = torch.zeros(T, device=device)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def compute_actor_loss(log_probs: torch.Tensor,
                       advantages: torch.Tensor) -> torch.Tensor:
    """
    REINFORCE Actor 损失（式 16）：
      L_actor = -E[ log π(α_t | s_t) * Adv(s_t, α_t) ]

    注意：最小化 L_actor 等价于最大化期望回报。

    Args:
        log_probs:  [T] 或 [T, B]  — log π(α_t | s_t)
        advantages: [T] 或 [T, B]  — Adv(s_t, α_t) = G_t - V(s_t)

    Returns:
        标量损失
    """
    # advantages detach：不让梯度通过优势函数流向 Critic
    return -(log_probs * advantages.detach()).mean()


def compute_critic_loss(values: torch.Tensor,
                        returns: torch.Tensor) -> torch.Tensor:
    """
    Critic MSE 损失（式 17）：
      L_critic = E[ (V(s_t) - G_t)² ]

    Args:
        values:  [T] 或 [T, B]  — Critic 的估计价值 V(s_t)
        returns: [T] 或 [T, B]  — 实际折扣回报 G_t

    Returns:
        标量损失
    """
    return F.mse_loss(values, returns.detach())
