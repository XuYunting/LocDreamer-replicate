"""
LocDreamer 配置文件
对应论文: "LocDreamer: World Model-Based Learning for Joint Indoor Tracking and Anchor Scheduling"

所有超参数均来自论文 Table 1 及正文描述。
"""

from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # =====================================================================
    # 环境配置
    # A: 目标（新）环境的锚点总数; Kt: 每步激活的锚点数
    # =====================================================================
    num_anchors: int = 5        # A — 新环境中锚点总数（Table 1）
    num_scheduled: int = 3      # Kt — 每时间步选取的锚点数（Table 1）
    num_source: int = 3         # |R| — 源环境预训练所用锚点数（论文实验设置）
    source_anchor_ids: tuple = (0, 5, 6)
    target_anchor_ids: tuple = (1, 2, 3, 4, 7)

    # 物理状态维度：z_t = [x, y, vx, vy]，Z=4（论文 System Model 节）
    state_dim: int = 4

    # =====================================================================
    # DSSM 架构参数
    # =====================================================================
    # H: RNN（序列模型）的隐藏状态维度（Table 1: H=50）
    hidden_dim: int = 50
    # RNN 层数（Table 1: 2层）
    rnn_layers: int = 2
    # Set Transformer 内部编码维度
    encoder_dim: int = 64
    # Set Transformer 多头注意力的头数
    num_heads: int = 4
    # MLP 隐藏层维度
    mlp_hidden: int = 64
    # 最小标准差，防止数值下溢
    min_std: float = 1e-4
    # 可选的 decoder 测距标准差上限。默认 None 表示完全按论文式(13b)
    # 学习标准差；设置后用于诊断 ELBO 是否靠过大方差解释距离。
    decoder_max_std: float = None
    # encoder 后验位置模式：
    #   mlp: 论文原始神经网络 encoder（默认，用于严格复现）
    #   wls: 额外加入测距几何 WLS 位置初始化（用于诊断/增强实验）
    encoder_mode: str = "mlp"
    # 是否仅对 encoder 的观测输入 (distance, anchor_x, anchor_y) 做尺度归一化。
    # 不使用位置标签、不改变 decoder 的米制物理距离，用于排查论文代码可能省略的实现细节。
    normalize_obs: bool = False
    env_width: float = 9.18
    env_height: float = 12.06
    # 后验状态用于重建/RNN 更新的方式：
    #   sample: 重参数化采样（默认）
    #   mean:   使用后验均值，仍为 ELBO-only，但降低采样噪声
    posterior_state_mode: str = "sample"

    # =====================================================================
    # 训练参数（Table 1）
    # =====================================================================
    dssm_epochs: int = 50       # Edssm — DSSM 预训练轮数
    imagine_epochs: int = 300   # Eimagine — WM 想象训练轮数（评估时用 1000）
    batch_size: int = 32        # B — 批大小
    seq_len: int = 32           # T — 序列长度
    lr: float = 1e-3            # 学习率（Adam）
    weight_decay: float = 1e-3  # 权重衰减
    pos_loss_weight: float = 0.0  # 位置辅助损失权重；默认关闭以匹配论文原始目标
    # Stage 1 时序展开后是否截断 z_t 到下一步的梯度。
    # 严格序列 ELBO 更接近不截断；保留开关便于对照已有结果。
    detach_state_between_steps: bool = True
    # 是否对物理不可能的 UWB range 做 mask 清洗：
    # encoder 输入用每个 anchor 的训练集中位数替换异常值，reconstruction loss
    # 只在有效 range 上计算。默认关闭，便于和原始读取方式对照。
    use_range_mask: bool = False
    min_valid_range: float = 0.0
    max_valid_range: float = 30.0

    # =====================================================================
    # 强化学习参数（Table 1）
    # =====================================================================
    gamma: float = 0.99         # γ — 折扣因子
    scheduler_pos_reward_weight: float = 0.0  # 调度奖励中几何定位一致性项权重（实验项）
    # Stage 2 想象训练策略：
    #   actor: 论文主方法，用 Actor-Critic 学调度
    #   random: LocDreamer-random baseline，想象训练时随机选 Kt 个 anchor
    #   all: LocDreamer-all(imagined) baseline，想象训练时使用全部目标 anchors
    stage2_policy: str = "actor"

    # =====================================================================
    # 物理运动模型参数
    # =====================================================================
    # 时间步长（秒），用于匀速运动模型：x_t = x_{t-1} + vx * dt
    dt: float = 0.1

    # =====================================================================
    # 数据与日志
    # =====================================================================
    data_path: str = "./data"           # 数据集目录
    checkpoint_dir: str = "./checkpoint_store/runs"
    log_interval: int = 10              # 每隔多少轮打印一次日志
    save_interval: int = 50             # 每隔多少轮保存一次检查点
    seed: int = 42

    # 自动选择 GPU/CPU
    device: str = field(
        default_factory=lambda: "cuda:1" if torch.cuda.is_available() else "cpu"
    )
