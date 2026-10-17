"""
Deep State Space Model (DSSM) — LocDreamer 的核心世界模型

对应论文 "Deep State Space Model in LocDreamer" 节及公式 (7)-(13)。

系统状态空间模型（SSM）：
  z_t ~ f_θz(z_{t-1}, h_t)        先验动态模型（式 3a 的深度版本）
  d_t ~ g_θd(z_t, h_t, α_t)       观测解码器（式 3b 的深度版本）
  h_t = r_θh(z_{t-1}^post, h_{t-1}) 确定性 RNN 隐藏状态（式 7）

四大组件：
  1. SequenceModel  — RNN，维护确定性历史记忆 h_t（式 7）
  2. DynamicModel   — 先验预测，结合物理运动模型与 MLP 残差（式 8-9）
  3. VariationalEncoder — 后验推断，基于 Set Transformer（式 10-11）
  4. DistanceDecoder — 距离重建，物理距离 + 可学习噪声模型（式 12-13）

训练目标（ELBO，式 6）：
  L_ELBO = Σ_t [ L_recon_t - L_dyn_t ]
  L_recon_t = E_{q_φ}[ Σ_{k∈Kt} log g_θd(d_t^k | z_t^post, h_t, p^k) ]
  L_dyn_t   = KL( q_φ(z_t^post | ...)  ||  f_θz(z_t^prior | ...) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

from .set_transformer import SetTransformerEncoder


# =========================================================================
# 辅助函数
# =========================================================================

def reparameterize(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    重参数化采样：z = μ + σ * ε，ε ~ N(0, I)
    使得梯度可以反向传播到 μ 和 σ。
    """
    eps = torch.randn_like(std)
    return mu + std * eps


def kl_gaussian(mu_post: torch.Tensor, std_post: torch.Tensor,
                mu_prior: torch.Tensor, std_prior: torch.Tensor) -> torch.Tensor:
    """
    两个对角高斯分布的 KL 散度（解析形式）：
      KL( N(μ₁,σ₁²) || N(μ₂,σ₂²) )
        = Σ_i [ log(σ₂_i/σ₁_i) + (σ₁_i² + (μ₁_i-μ₂_i)²)/(2σ₂_i²) - 1/2 ]

    Returns:
        [B, Z] 各维度的 KL 散度（对 batch 维度保留，后续求和/均值）
    """
    var_post = std_post.pow(2)
    var_prior = std_prior.pow(2)
    kl = (
        torch.log(std_prior / (std_post + 1e-8))
        + (var_post + (mu_post - mu_prior).pow(2)) / (2.0 * var_prior + 1e-8)
        - 0.5
    )
    return kl  # [B, Z]


# =========================================================================
# 1. 序列模型：RNN（确定性隐藏状态 h_t）
# =========================================================================

class SequenceModel(nn.Module):
    """
    序列模型（式 7）：h_t = r_θh(z_{t-1}^posterior, h_{t-1})

    使用 GRU 作为循环函数 r_θh(·)。
    h_t 是确定性的，扮演"记忆"角色，编码了历史轨迹的非马尔可夫依赖。

    Args:
        input_dim:  输入维度 = 状态维度 Z=4（上一步后验状态）
        hidden_dim: GRU 隐藏状态维度 H=50（Table 1）
        num_layers: GRU 层数（Table 1: 2层）
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 多层 GRU：input_size=Z, hidden_size=H, batch_first=True
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,   # 输入形状 [B, 1, Z]
        )

    def forward(self, z_post_prev: torch.Tensor,
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_post_prev: [B, Z]  — 上一步后验状态 z_{t-1}^posterior
            h_prev:      [num_layers, B, H]  — 上一步 GRU 隐藏状态

        Returns:
            h_t: [B, H]  — 更新后的 GRU 输出（取最后一层）
            h_state: [num_layers, B, H]  — 更新后的完整隐藏状态（用于下一步）
        """
        # 添加时间维度：[B, Z] → [B, 1, Z]
        x = z_post_prev.unsqueeze(1)
        # GRU 前向传播
        out, h_state = self.gru(x, h_prev)  # out: [B, 1, H]
        h_t = out.squeeze(1)                 # [B, H]
        return h_t, h_state

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """初始化零隐藏状态"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


# =========================================================================
# 2. 动态模型：先验预测（物理模型 + MLP 残差）
# =========================================================================

class DynamicModel(nn.Module):
    """
    先验动态模型（式 8-9）：
      z_t^prior ~ N(μ_z,t^prior, diag(σ_z,t^prior)²)
      [μ_z,t^prior, σ_z,t^prior] = f_θz(z_{t-1}^post, h_t)
                                  = f_phy(z_{t-1}^post) + MLP_θz(z_{t-1}^post, h_t)

    f_phy 是匀速运动模型（Constant Velocity Model，Wang et al. 2025a）：
      x_t  = x_{t-1}  + v_x * dt
      y_t  = y_{t-1}  + v_y * dt
      v_x,t = v_{x,t-1}
      v_y,t = v_{y,t-1}

    MLP_θz 学习残差，补偿模型失配（如加速、转弯等非匀速运动）。

    Args:
        state_dim:  状态维度 Z=4
        hidden_dim: RNN 隐藏维度 H=50
        mlp_hidden: MLP 内部隐藏维度
        dt:         时间步长（秒）
        min_std:    标准差的下限
    """

    def __init__(self, state_dim: int, hidden_dim: int,
                 mlp_hidden: int = 64, dt: float = 0.1, min_std: float = 1e-4):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        self.min_std = min_std

        # 构造匀速运动转移矩阵 F，状态 [x, y, vx, vy]：
        #   x_t  = x_{t-1} + vx * dt
        #   y_t  = y_{t-1} + vy * dt
        #   vx_t = vx_{t-1}
        #   vy_t = vy_{t-1}
        F = torch.eye(state_dim)
        F[0, 2] = dt   # x += vx * dt
        F[1, 3] = dt   # y += vy * dt
        # register_buffer 使其随模型移动到 GPU，但不参与梯度计算
        self.register_buffer('F', F)

        # MLP 学习残差均值和对数标准差，输入 [z_{t-1}^post, h_t]，输出 [Δμ, log_σ]
        # 输入维度 = Z + H = 4 + 50 = 54
        input_dim = state_dim + hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ELU(),
            # 输出 2*Z：前 Z 个为均值残差 Δμ，后 Z 个为 log_σ
            nn.Linear(mlp_hidden, 2 * state_dim),
        )

    def physics_predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        匀速运动模型：f_phy(z_{t-1}) = F @ z_{t-1}

        Args:
            z: [B, Z]  — 上一步状态
        Returns:
            [B, Z]  — 物理模型预测的均值
        """
        # F: [Z, Z]，z: [B, Z] → [B, Z]
        return z @ self.F.t()

    def forward(self, z_post_prev: torch.Tensor,
                h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_post_prev: [B, Z]  — 上一步后验状态 z_{t-1}^posterior
            h_t:         [B, H]  — 当前 RNN 隐藏状态（已由序列模型更新）

        Returns:
            mu_prior:  [B, Z]  — 先验均值
            std_prior: [B, Z]  — 先验标准差（正数）
        """
        # 物理模型提供均值的初始预测
        mu_phy = self.physics_predict(z_post_prev)  # [B, Z]

        # MLP 学习残差，输入为 [z_{t-1}^post, h_t]
        mlp_in = torch.cat([z_post_prev, h_t], dim=-1)   # [B, Z+H]
        mlp_out = self.mlp(mlp_in)                        # [B, 2Z]

        delta_mu = mlp_out[..., :self.state_dim]          # [B, Z] 均值残差
        log_std  = mlp_out[..., self.state_dim:]          # [B, Z] 对数标准差

        # 最终先验均值 = 物理预测 + 数据驱动残差（式 9）
        mu_prior  = mu_phy + delta_mu                     # [B, Z]
        # 标准差 = softplus(log_std) + min_std，保证正数且有下界
        std_prior = F.softplus(log_std) + self.min_std    # [B, Z]

        return mu_prior, std_prior


# =========================================================================
# 3. 变分编码器：后验推断（Set Transformer + MLP）
# =========================================================================

class VariationalEncoder(nn.Module):
    """
    变分编码器（式 10-11）：
      z_t^posterior ~ N(μ_z,t^posterior, diag(σ_z,t^posterior)²)
      [μ_z,t^posterior, σ_z,t^posterior] = q_φ(z_{t-1}^post, h_t, o(α_t))

    其中 o(α_t) = {(d_t^k, p^k) | α_t^k = 1} 是激活锚点的观测集合。
    使用 Set Transformer 对 o(α_t) 进行排列不变编码。

    Args:
        state_dim:   状态维度 Z=4
        hidden_dim:  RNN 隐藏维度 H=50
        encoder_dim: Set Transformer 内部维度
        num_heads:   Set Transformer 注意力头数
        mlp_hidden:  MLP 隐藏层维度
        min_std:     标准差下限
    """

    def __init__(self, state_dim: int, hidden_dim: int,
                 encoder_dim: int = 64, num_heads: int = 4,
                 mlp_hidden: int = 64, min_std: float = 1e-4,
                 encoder_mode: str = "mlp",
                 normalize_obs: bool = False,
                 env_width: float = 9.18,
                 env_height: float = 12.06):
        super().__init__()
        self.min_std = min_std
        if encoder_mode not in {"mlp", "wls"}:
            raise ValueError(f"encoder_mode 必须是 'mlp' 或 'wls'，当前为: {encoder_mode}")
        self.encoder_mode = encoder_mode
        self.normalize_obs = normalize_obs
        self.env_width = env_width
        self.env_height = env_height
        self.distance_scale = (env_width ** 2 + env_height ** 2) ** 0.5

        # Set Transformer：对观测集合 {(d_t^k, p^k)} 进行排列不变编码
        # 输入每个元素维度 = 3：(distance, anchor_x, anchor_y)
        self.set_encoder = SetTransformerEncoder(
            input_dim=3,
            d_model=encoder_dim,
            num_heads=num_heads,
            num_sab_layers=2,
        )

        # MLP 将上下文向量 [z_{t-1}^post, h_t, set_out] 映射到 [μ, log_σ]
        # 输入维度 = Z + H + encoder_dim = 4 + 50 + 64 = 118
        context_dim = state_dim + hidden_dim + encoder_dim
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 2 * state_dim),
        )

    def normalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Scale encoder observations to comparable numeric ranges."""
        if not self.normalize_obs:
            return obs
        scale = obs.new_tensor([self.distance_scale, self.env_width, self.env_height])
        return obs / scale

    @staticmethod
    def wls_position(obs: torch.Tensor) -> torch.Tensor:
        """
        从一组距离观测做二维线性最小二乘定位。

        obs[b, k] = [d_k, anchor_x, anchor_y]。以第一个锚点为参考：
          2(p_k - p_0)^T x = d_0^2 - d_k^2 + ||p_k||^2 - ||p_0||^2
        这给 encoder 一个物理几何起点，网络只需学习 NLoS/噪声残差。
        """
        d = obs[..., 0]       # [B, K]
        p = obs[..., 1:3]     # [B, K, 2]
        p0 = p[:, :1, :]      # [B, 1, 2]
        d0 = d[:, :1]         # [B, 1]

        A = 2.0 * (p[:, 1:, :] - p0)  # [B, K-1, 2]
        b = (
            d0.pow(2)
            - d[:, 1:].pow(2)
            + p[:, 1:, :].pow(2).sum(dim=-1)
            - p0.pow(2).sum(dim=-1)
        )  # [B, K-1]

        # torch.linalg.lstsq 支持 batch，K=3 时正好是 2x2，K>3 为超定。
        sol = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
        sol_x = sol[..., 0].clamp(0.0, 9.18)
        sol_y = sol[..., 1].clamp(0.0, 12.06)
        sol = torch.stack([sol_x, sol_y], dim=-1)
        return sol

    def forward(self, z_post_prev: torch.Tensor, h_t: torch.Tensor,
                obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_post_prev: [B, Z]    — 上一步后验状态
            h_t:         [B, H]    — 当前 RNN 隐藏状态
            obs:         [B, K, 3] — 激活锚点的观测集合
                                     obs[:, k, :] = [d_t^k, p_x^k, p_y^k]

        Returns:
            mu_post:  [B, Z]  — 后验均值
            std_post: [B, Z]  — 后验标准差（正数）
        """
        # Set Transformer 编码观测集合，输出排列不变的聚合向量
        set_out = self.set_encoder(self.normalize_observation(obs))    # [B, encoder_dim]

        # 拼接所有上下文信息
        context = torch.cat([z_post_prev, h_t, set_out], dim=-1)  # [B, Z+H+D]

        # MLP 推断后验分布参数
        out = self.mlp(context)            # [B, 2Z]

        mu_post  = out[..., :z_post_prev.shape[-1]]   # [B, Z]
        log_std  = out[..., z_post_prev.shape[-1]:]   # [B, Z]
        std_post = F.softplus(log_std) + self.min_std  # [B, Z]

        if self.encoder_mode == "wls":
            # 几何增强实验：位置均值锚定到 WLS 解；速度仍由 MLP 推断。
            pos_wls = self.wls_position(obs)
            mu_post = torch.cat([pos_wls, mu_post[..., 2:]], dim=-1)

        return mu_post, std_post


# =========================================================================
# 4. 距离解码器：重建观测（物理距离 + 可学习噪声）
# =========================================================================

class DistanceDecoder(nn.Module):
    """
    距离解码器（式 12-13）：
      d̂_t^k ~ N(μ_d,t^k, (σ_d,t^k)²)
      μ_d,t^k = ||p̃_t^posterior - p^k||         （物理距离，式 13a）
      σ_d,t^k = MLP_θd(z̃_t^posterior, h_t, p^k) （可学习噪声，式 13b）

    物理距离模型（式 1）：d_t^k = ||p_t - p^k|| + n_t^k
    均值通过直接计算欧氏距离得到（无需训练），MLP 学习噪声模型。

    Args:
        state_dim:  状态维度 Z=4，z[:2] 为位置 [x, y]
        hidden_dim: RNN 隐藏维度 H=50
        mlp_hidden: MLP 隐藏层维度
        min_std:    噪声标准差下限
    """

    def __init__(self, state_dim: int, hidden_dim: int,
                 mlp_hidden: int = 64, min_std: float = 1e-4,
                 max_std: float = None):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        if max_std is not None and max_std <= min_std:
            raise ValueError(f"decoder max_std={max_std} 必须大于 min_std={min_std}")

        # MLP 学习测量噪声标准差
        # 输入：[z̃_t^post(Z), h_t(H), p^k(2)]，输出：log_σ (1个标量)
        input_dim = state_dim + hidden_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, z_sample: torch.Tensor, h_t: torch.Tensor,
                anchor_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为单个锚点（或批量锚点）计算距离分布参数。

        Args:
            z_sample:   [B, Z]    — 重参数化采样 z̃_t^posterior（或 z̃_t^prior）
            h_t:        [B, H]    — RNN 隐藏状态
            anchor_pos: [B, 2] 或 [2]  — 锚点位置 p^k = [px, py]

        Returns:
            mu_d:  [B, 1]  — 距离均值（物理距离）
            std_d: [B, 1]  — 距离标准差（可学习噪声）
        """
        # 从状态中提取位置分量（前两维）
        pos_estimate = z_sample[..., :2]   # [B, 2]

        # anchor_pos 可能是 [2]，需广播
        if anchor_pos.dim() == 1:
            anchor_pos = anchor_pos.unsqueeze(0).expand_as(pos_estimate)

        # 物理距离均值：||p̃ - p^k|| + epsilon（防止 0 点的梯度问题）
        diff = pos_estimate - anchor_pos                            # [B, 2]
        mu_d = (diff.pow(2).sum(dim=-1, keepdim=True) + 1e-8).sqrt()  # [B, 1]

        # MLP 学习噪声标准差
        mlp_in = torch.cat([z_sample, h_t, anchor_pos], dim=-1)    # [B, Z+H+2]
        log_std = self.mlp(mlp_in)                                  # [B, 1]
        if self.max_std is None:
            std_d = F.softplus(log_std) + self.min_std              # [B, 1]
        else:
            std_d = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(log_std)

        return mu_d, std_d

    def log_prob(self, d_obs: torch.Tensor, mu_d: torch.Tensor,
                 std_d: torch.Tensor) -> torch.Tensor:
        """
        计算观测距离在高斯分布下的对数概率（重建损失的主要项）：
          log N(d; μ, σ²) = -0.5*log(2π) - log(σ) - (d-μ)² / (2σ²)

        Args:
            d_obs: [B, 1]  — 观测（真实或想象）距离
            mu_d:  [B, 1]  — 均值
            std_d: [B, 1]  — 标准差

        Returns:
            [B, 1]  — 对数概率
        """
        dist = torch.distributions.Normal(mu_d, std_d)
        return dist.log_prob(d_obs)  # [B, 1]


# =========================================================================
# 5. 完整 DSSM（整合四个组件）
# =========================================================================

class DSSM(nn.Module):
    """
    Deep State Space Model — LocDreamer 的世界模型核心

    整合四个组件实现端到端的概率跟踪：
      - SequenceModel  (h_t 更新)
      - DynamicModel   (先验 z_t^prior)
      - VariationalEncoder (后验 z_t^posterior)
      - DistanceDecoder (重建 d̂_t^k)

    训练目标：最大化 ELBO（式 6）：
      L_ELBO = Σ_t [ L_recon_t - L_dyn_t ]
    """

    def __init__(self, state_dim: int = 4, hidden_dim: int = 50,
                 rnn_layers: int = 2, encoder_dim: int = 64,
                 num_heads: int = 4, mlp_hidden: int = 64,
                 dt: float = 0.1, min_std: float = 1e-4,
                 encoder_mode: str = "mlp",
                 posterior_state_mode: str = "sample",
                 normalize_obs: bool = False,
                 env_width: float = 9.18,
                 env_height: float = 12.06,
                 decoder_max_std: float = None):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        if posterior_state_mode not in {"sample", "mean"}:
            raise ValueError(
                f"posterior_state_mode 必须是 'sample' 或 'mean'，当前为: {posterior_state_mode}"
            )
        self.posterior_state_mode = posterior_state_mode

        # 四个核心组件
        self.sequence_model = SequenceModel(state_dim, hidden_dim, rnn_layers)
        self.dynamic_model = DynamicModel(state_dim, hidden_dim, mlp_hidden, dt, min_std)
        self.encoder = VariationalEncoder(
            state_dim, hidden_dim, encoder_dim, num_heads, mlp_hidden, min_std,
            encoder_mode=encoder_mode,
            normalize_obs=normalize_obs,
            env_width=env_width,
            env_height=env_height,
        )
        self.decoder = DistanceDecoder(
            state_dim, hidden_dim, mlp_hidden, min_std, max_std=decoder_max_std
        )

    def init_state(self, batch_size: int,
                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化状态：z_0^posterior 和 GRU 隐藏状态

        Returns:
            z_post: [B, Z]  — 全零初始后验状态
            h_gru:  [num_layers, B, H]  — 全零 GRU 隐藏状态
        """
        z_post = torch.zeros(batch_size, self.state_dim, device=device)
        h_gru = self.sequence_model.init_hidden(batch_size, device)
        return z_post, h_gru

    def step(self, z_post_prev: torch.Tensor, h_gru_prev: torch.Tensor,
             obs: torch.Tensor, anchor_positions: torch.Tensor
             ) -> Dict[str, torch.Tensor]:
        """
        单步前向传播（用于预训练阶段，输入真实观测）

        完整流程（Algorithm 1, Lines 5-10）：
          1. 更新 RNN 隐藏状态 h_t
          2. 预测先验 z_t^prior
          3. 推断后验 z_t^posterior
          4. 重建距离 d̂_t^k

        Args:
            z_post_prev:      [B, Z]    — 上一步后验状态
            h_gru_prev:       [L, B, H] — 上一步 GRU 隐藏状态
            obs:              [B, K, 3] — 当前步的观测集合（真实或想象）
            anchor_positions: [K, 2]    — 当前步活跃锚点的位置

        Returns:
            字典包含：
              'mu_prior', 'std_prior'     — 先验参数
              'mu_post', 'std_post'       — 后验参数
              'z_post_sample'             — 后验重参数化样本
              'h_t'                       — 新 RNN 输出
              'h_gru'                     — 新完整 GRU 隐藏状态
        """
        B = z_post_prev.shape[0]

        # 当前 GRU 状态即 Algorithm 1 中的 h_t。先用 h_t 做先验/后验推断，
        # 再用当前后验 z_t 更新到 h_{t+1}。
        h_t = h_gru_prev[-1]  # [B, H]

        # ── 步骤 1：先验预测 z_t^prior（式 8-9）──
        mu_prior, std_prior = self.dynamic_model(z_post_prev, h_t)  # [B, Z]

        # ── 步骤 2：后验推断 z_t^posterior（式 10-11）──
        mu_post, std_post = self.encoder(z_post_prev, h_t, obs)     # [B, Z]

        # 后验状态用于解码器和 RNN 更新。默认重参数化采样；mean 模式仍然
        # 是 ELBO-only，不引入位置标签或几何先验，只降低 Monte Carlo 噪声。
        if self.posterior_state_mode == "mean":
            z_post_sample = mu_post
        else:
            z_post_sample = reparameterize(mu_post, std_post)        # [B, Z]

        # ── 步骤 3：更新 RNN 隐藏状态 h_{t+1}（Algorithm 1, Line 8）──
        _, h_gru = self.sequence_model(z_post_sample, h_gru_prev)

        return {
            'mu_prior': mu_prior,
            'std_prior': std_prior,
            'mu_post': mu_post,
            'std_post': std_post,
            'z_post_sample': z_post_sample,
            'h_t': h_t,
            'h_gru': h_gru,
        }

    def decode_distance(self, z_sample: torch.Tensor, h_t: torch.Tensor,
                        anchor_positions: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为一组锚点批量计算重建距离分布（式 12-13）

        Args:
            z_sample:         [B, Z]  — 后验重参数化样本
            h_t:              [B, H]  — RNN 隐藏状态
            anchor_positions: [K, 2] 或 [B, K, 2]  — 锚点位置

        Returns:
            mu_d:  [B, K]  — 各锚点距离均值
            std_d: [B, K]  — 各锚点距离标准差
        """
        B = z_sample.shape[0]
        K = anchor_positions.shape[-2] if anchor_positions.dim() > 1 else 1

        # 将所有锚点批量处理
        if anchor_positions.dim() == 2:
            # [K, 2] → 对每个锚点计算，结果堆叠
            mu_list, std_list = [], []
            for k in range(anchor_positions.shape[0]):
                p_k = anchor_positions[k].unsqueeze(0).expand(B, -1)  # [B, 2]
                mu_k, std_k = self.decoder(z_sample, h_t, p_k)        # [B, 1]
                mu_list.append(mu_k)
                std_list.append(std_k)
            mu_d  = torch.cat(mu_list, dim=-1)   # [B, K]
            std_d = torch.cat(std_list, dim=-1)   # [B, K]
        else:
            raise ValueError(f"anchor_positions 应为 [K,2]，当前形状: {anchor_positions.shape}")

        return mu_d, std_d

    def compute_elbo(self, d_obs: torch.Tensor,
                     mu_prior: torch.Tensor, std_prior: torch.Tensor,
                     mu_post: torch.Tensor, std_post: torch.Tensor,
                     z_post_sample: torch.Tensor, h_t: torch.Tensor,
                     anchor_positions: torch.Tensor,
                     obs_mask: torch.Tensor = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算单步 ELBO（式 6）：
          L_ELBO_t = L_recon_t - L_dyn_t

        Args:
            d_obs:            [B, K]    — 观测距离（真实或想象）
            mu_prior/std_prior: [B, Z] — 先验参数
            mu_post/std_post:   [B, Z] — 后验参数
            z_post_sample:    [B, Z]   — 后验重参数化样本
            h_t:              [B, H]   — RNN 输出
            anchor_positions: [K, 2]   — 锚点位置

        Returns:
            l_recon: 标量  — 重建损失（均值）
            l_dyn:   标量  — 动态损失/KL 散度（均值）
        """
        # ── 重建损失 L_recon（ELBO 中的正项）──
        # 计算每个锚点的重建分布参数
        mu_d, std_d = self.decode_distance(z_post_sample, h_t, anchor_positions)
        # [B, K]，对数概率
        log_p = self.decoder.log_prob(
            d_obs.unsqueeze(-1) if d_obs.dim() == 1 else d_obs.unsqueeze(-1),
            mu_d.unsqueeze(-1),
            std_d.unsqueeze(-1)
        ).squeeze(-1)

        # 更简洁的写法：逐锚点计算
        B, K = d_obs.shape
        if obs_mask is None:
            obs_mask = torch.ones(B, K, device=d_obs.device)
        else:
            obs_mask = obs_mask.to(d_obs.device).float()
        log_p_all = torch.zeros(B, K, device=d_obs.device)
        for k in range(K):
            p_k = anchor_positions[k].unsqueeze(0).expand(B, -1)    # [B, 2]
            mu_k, std_k = self.decoder(z_post_sample, h_t, p_k)     # [B, 1]
            log_p_all[:, k] = self.decoder.log_prob(
                d_obs[:, k:k+1], mu_k, std_k
            ).squeeze(-1)
        # L_recon_t = E_{q_φ}[Σ_{k∈Kt} log g(d_t^k | ...)]。
        # 启用 range mask 时，物理不可能的测距不参与重建项，避免模型
        # 被失败测距输出带偏；不启用时 obs_mask 全 1，行为与原来一致。
        l_recon = (log_p_all * obs_mask).sum(dim=-1).mean()  # 标量

        # ── 动态损失 L_dyn（ELBO 中的负项，KL 散度）──
        # KL(q_φ(z^post | ...) || f_θz(z^prior | ...))
        kl = kl_gaussian(mu_post, std_post, mu_prior, std_prior)   # [B, Z]
        l_dyn = kl.sum(dim=-1).mean()  # 对状态维度求和，对 batch 取均值

        return l_recon, l_dyn

    def imagine(self, z_post_prev: torch.Tensor, h_gru_prev: torch.Tensor,
                all_anchor_positions: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """
        想象步骤（Algorithm 1, Lines 15-16）：
        使用先验状态生成对所有锚点的想象测量（无需真实观测）。

        具体流程：
          1. 读取当前隐藏状态 h_t
          2. 从先验预测 z_t^prior（不用后验，因为没有真实观测）
          3. 用先验样本生成所有锚点的想象距离测量

        Args:
            z_post_prev:        [B, Z]   — 上一步后验状态
            h_gru_prev:         [L, B, H] — 上一步 GRU 隐藏状态
            all_anchor_positions: [A, 2] — 所有锚点位置

        Returns:
            字典包含：
              'mu_prior', 'std_prior'   — 先验参数
              'z_prior_sample'          — 先验重参数化样本
              'h_t'                     — RNN 输出
              'h_gru'                   — 当前 GRU 隐藏状态（后验更新后再推进）
              'imagined_distances'      — [B, A] 想象的距离测量
              'imagined_stds'           — [B, A] 对应的不确定性
        """
        # Algorithm 1 Line 15 使用当前 h_t；Line 20 才更新到 h_{t+1}。
        h_t = h_gru_prev[-1]

        # ── 步骤 2：先验预测 ──
        mu_prior, std_prior = self.dynamic_model(z_post_prev, h_t)
        z_prior_sample = reparameterize(mu_prior, std_prior)   # [B, Z]

        # ── 步骤 3：用先验样本生成所有锚点的想象距离（式 13，用 z_prior）──
        mu_d, std_d = self.decode_distance(z_prior_sample, h_t, all_anchor_positions)
        # 从高斯分布采样想象距离（reparameterize）
        imagined_distances = reparameterize(mu_d, std_d)       # [B, A]

        return {
            'mu_prior': mu_prior,
            'std_prior': std_prior,
            'z_prior_sample': z_prior_sample,
            'h_t': h_t,
            'h_gru': h_gru_prev,
            'imagined_distances': imagined_distances,
            'imagined_stds': std_d,
        }

    def update_posterior(self, z_post_prev: torch.Tensor, h_t: torch.Tensor,
                         h_gru: torch.Tensor,
                         imagined_distances: torch.Tensor,
                         selected_anchor_idx: torch.Tensor,
                         all_anchor_positions: torch.Tensor
                         ) -> Dict[str, torch.Tensor]:
        """
        用调度器选择的锚点更新后验（Algorithm 1, Lines 17-20）。

        Args:
            z_post_prev:         [B, Z]   — 上一步后验
            h_t:                 [B, H]   — 当前 RNN 输出
            h_gru:               [L, B, H] — 当前 GRU 隐藏状态
            imagined_distances:  [B, A]   — 所有锚点的想象距离
            selected_anchor_idx: [B, Kt]  — 被选中锚点的索引（int）
            all_anchor_positions: [A, 2]  — 所有锚点位置

        Returns:
            字典包含后验参数和更新的隐藏状态
        """
        B, Kt = selected_anchor_idx.shape
        device = z_post_prev.device

        # 构造观测集合 o(α_t) = {(d_t^k, p^k) | α_t^k = 1}，形状 [B, Kt, 3]
        obs_list = []
        for b in range(B):
            obs_b = []
            for k_idx in selected_anchor_idx[b]:
                k = k_idx.item()
                d_k = imagined_distances[b, k].unsqueeze(0)        # [1]
                p_k = all_anchor_positions[k]                       # [2]
                obs_b.append(torch.cat([d_k, p_k], dim=-1))        # [3]
            obs_list.append(torch.stack(obs_b, dim=0))             # [Kt, 3]
        obs = torch.stack(obs_list, dim=0).to(device)              # [B, Kt, 3]

        # 后验推断（式 11）
        mu_post, std_post = self.encoder(z_post_prev, h_t, obs)
        if self.posterior_state_mode == "mean":
            z_post_sample = mu_post
        else:
            z_post_sample = reparameterize(mu_post, std_post)

        # 更新 RNN 隐藏状态（式 7）：h_{t+1} = r_θh(z_t^post, h_t)
        _, h_gru_next = self.sequence_model(z_post_sample, h_gru)

        return {
            'mu_post': mu_post,
            'std_post': std_post,
            'z_post_sample': z_post_sample,
            'h_gru': h_gru_next,
            'obs': obs,
        }

    def get_position_estimate(self, z_sample: torch.Tensor) -> torch.Tensor:
        """
        从状态样本中提取位置估计（前两维）

        Args:
            z_sample: [B, Z] 或 [Z]

        Returns:
            [B, 2] 或 [2]  — 估计位置 [x, y]
        """
        return z_sample[..., :2]
