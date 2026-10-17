"""
LocDreamer 评估脚本

复现论文 Table 2 中的六种对比方法：
  1. EKF - random scheduling      扩展卡尔曼滤波 + 随机选3个锚点
  2. EKF - all anchors            扩展卡尔曼滤波 + 全部5个锚点
  3. DSSM - all anchors (real)    DSSM + 全部锚点真实数据训练
  4. LocDreamer - random          DSSM(想象) + 随机调度
  5. LocDreamer - all (imagined)  DSSM(想象) + 全部锚点
  6. LocDreamer - scheduling      DSSM(想象) + AC 调度器（论文主方法）

评估指标（Table 2）：
  MAE | RMSE | 50th 百分位误差 | 90th 百分位误差（单位：米）
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import random

from config import Config
from models.dssm import DSSM, reparameterize
from models.actor_critic import Actor, Critic
from utils.dataset import get_data_loaders
from utils.metrics import compute_all_metrics, print_metrics
from train_init import build_observation_set, load_checkpoint
from train_init import parse_anchor_ids


# =========================================================================
# 辅助：EKF（扩展卡尔曼滤波）基线
# =========================================================================

class ExtendedKalmanFilter:
    """
    扩展卡尔曼滤波器（EKF）用于 UWB 测距跟踪

    对应论文基线方法（式 1 的经典实现），使用匀速运动模型。

    状态向量：z = [x, y, vx, vy]，维度 Z=4
    过程噪声：σ_acc = 0.2 m/s²（Table 2 注释）
    测量噪声：σ_n = 1.0 m（Table 2 注释）
    """

    def __init__(self, dt: float = 0.1,
                 sigma_acc: float = 0.2, sigma_n: float = 1.0):
        self.dt = dt
        self.Z = 4   # 状态维度

        # 状态转移矩阵 F（匀速运动）
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # 过程噪声协方差 Q（离散时间 CV 模型）
        # σ_acc² * [[dt^4/4, 0, dt^3/2, 0], ...]
        q = sigma_acc ** 2
        self.Q = q * np.array([
            [dt**4/4, 0,       dt**3/2, 0],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0],
            [0,       dt**3/2, 0,       dt**2],
        ], dtype=np.float64)

        # 测量噪声方差
        self.sigma_n = sigma_n

        # 状态和协方差
        self.x = None    # [Z]
        self.P = None    # [Z, Z]

    def init(self, pos_init: np.ndarray = None):
        """初始化状态"""
        if pos_init is not None:
            self.x = np.array([pos_init[0], pos_init[1], 0.0, 0.0])
        else:
            self.x = np.zeros(self.Z)
        self.P = np.eye(self.Z) * 10.0   # 较大初始不确定性

    def predict(self):
        """EKF 预测步骤：x = F*x，P = F*P*F^T + Q"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, distances: np.ndarray, anchor_positions: np.ndarray):
        """
        EKF 更新步骤（逐锚点串行更新）

        Args:
            distances:        [K]  — K 个活跃锚点的距离测量
            anchor_positions: [K, 2] — 对应锚点位置
        """
        for k in range(len(distances)):
            d_k   = distances[k]
            p_k   = anchor_positions[k]
            pos   = self.x[:2]

            # 观测函数（非线性）：h(x) = ||pos - p^k||
            diff  = pos - p_k
            dist  = np.linalg.norm(diff) + 1e-8

            # 观测雅可比 H：∂h/∂x = [Δx/d, Δy/d, 0, 0]
            H = np.array([[diff[0]/dist, diff[1]/dist, 0, 0]])   # [1, Z]
            # 卡尔曼增益
            R = np.array([[self.sigma_n ** 2]])
            S = H @ self.P @ H.T + R                              # [1, 1]
            K = self.P @ H.T @ np.linalg.inv(S)                   # [Z, 1]
            # 更新
            innovation = d_k - dist
            self.x = self.x + (K @ [[innovation]]).squeeze()
            self.P = (np.eye(self.Z) - K @ H) @ self.P

    def get_position(self) -> np.ndarray:
        return self.x[:2].copy()


def linear_ls_position(distances: np.ndarray, anchor_positions: np.ndarray,
                       env_width: float = 9.18,
                       env_height: float = 12.06) -> np.ndarray:
    """
    用单帧测距和 anchor 坐标做二维线性最小二乘初始化。

    只使用测距与 anchor 坐标，不使用真实位置标签。返回 [x, y]。
    """
    distances = np.asarray(distances, dtype=np.float64)
    anchor_positions = np.asarray(anchor_positions, dtype=np.float64)
    valid = np.isfinite(distances) & (distances > 0.0) & (distances < 30.0)
    if valid.sum() < 3:
        return np.array([0.0, 0.0], dtype=np.float64)

    d = distances[valid]
    p = anchor_positions[valid]
    p0 = p[:1]
    d0 = d[:1]

    A = 2.0 * (p[1:] - p0)
    b = (
        d0 ** 2
        - d[1:] ** 2
        + np.sum(p[1:] ** 2, axis=-1)
        - np.sum(p0 ** 2, axis=-1)
    )
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    sol[0] = np.clip(sol[0], 0.0, env_width)
    sol[1] = np.clip(sol[1], 0.0, env_height)
    return sol


def evaluate_ekf(data_loader, anchor_positions: np.ndarray,
                 num_scheduled: int = 3,
                 all_anchors: bool = False,
                 init_mode: str = 'gt',
                 env_width: float = 9.18,
                 env_height: float = 12.06) -> Dict[str, float]:
    """
    评估 EKF 基线（对应 Table 2 前两行）

    Args:
        data_loader:    目标环境数据加载器
        anchor_positions: [A, 2]，锚点位置（numpy）
        num_scheduled:  Kt，每步激活锚点数
        all_anchors:    True → 使用全部锚点（EKF-all），False → 随机调度
        init_mode:      zero → 零初始化；center → 房间中心初始化；
                        random → 房间范围内随机初始化；
                        ls → 第一帧测距最小二乘初始化；
                        gt → 第一帧真实位置初始化

    Returns:
        评估指标字典
    """
    A = len(anchor_positions)
    ekf = ExtendedKalmanFilter()

    all_pred = []
    all_gt   = []

    for batch in data_loader:
        distances = batch['distances'].numpy()   # [B, T, A]
        positions = batch['positions'].numpy()   # [B, T, 2]
        B, T, _ = distances.shape

        for b in range(B):
            if init_mode == 'zero':
                ekf.init(None)
            elif init_mode == 'center':
                ekf.init(np.array([env_width / 2.0, env_height / 2.0]))
            elif init_mode == 'random':
                ekf.init(np.array([
                    np.random.uniform(0.0, env_width),
                    np.random.uniform(0.0, env_height),
                ]))
            elif init_mode == 'ls':
                pos_init = linear_ls_position(
                    distances[b, 0], anchor_positions,
                    env_width=env_width, env_height=env_height
                )
                ekf.init(pos_init)
            elif init_mode == 'gt':
                ekf.init(positions[b, 0])
            else:
                raise ValueError(f"未知 EKF 初始化方式：{init_mode}")

            for t in range(T):
                ekf.predict()

                if all_anchors:
                    active_idx = list(range(A))
                else:
                    # 随机选 Kt 个锚点
                    active_idx = sorted(random.sample(range(A), num_scheduled))

                ekf.update(
                    distances[b, t, active_idx],
                    anchor_positions[active_idx]
                )

                all_pred.append(ekf.get_position())
                all_gt.append(positions[b, t])

    return compute_all_metrics(
        np.array(all_pred), np.array(all_gt)
    )


# =========================================================================
# DSSM/LocDreamer 评估
# =========================================================================

@torch.no_grad()
def evaluate_dssm(dssm: DSSM,
                  data_loader,
                  anchor_positions: torch.Tensor,
                  device: torch.device,
                  actor: Optional[Actor] = None,
                  num_scheduled: int = 3,
                  mode: str = 'all'
                  ) -> Dict[str, float]:
    """
    评估 DSSM/LocDreamer（对应 Table 2 后四行）

    模式：
      'all'       → 使用全部 A 个锚点（LocDreamer-all 或 DSSM-all）
      'random'    → 随机选 Kt 个锚点（LocDreamer-random）
      'scheduled' → 调度选择 Kt 个锚点（LocDreamer-scheduling，主方法）

    Args:
        dssm:             DSSM 模型（已训练）
        data_loader:      目标环境数据加载器（真实测量）
        anchor_positions: [A, 2]，锚点位置
        device:           计算设备
        actor:            Actor 模型（仅 mode='scheduled' 时需要）
        num_scheduled:    Kt
        mode:             评估模式

    Returns:
        评估指标字典
    """
    dssm.eval()
    if actor is not None:
        actor.eval()

    A = anchor_positions.shape[0]
    all_pred = []
    all_gt   = []

    for batch in data_loader:
        distances = batch['distances'].to(device)   # [B, T, A]
        positions = batch['positions'].to(device)   # [B, T, 2]
        B, T, _ = distances.shape

        # 初始化状态
        z_post, h_gru = dssm.init_state(B, device)

        for t in range(T):
            d_t = distances[:, t, :]   # [B, A]

            if mode == 'all':
                # 使用全部锚点
                obs_t = build_observation_set(d_t, anchor_positions, B, device)
                step_out = dssm.step(z_post, h_gru, obs_t, anchor_positions)

            elif mode == 'random':
                # 随机选 Kt 个锚点
                active_idx = sorted(random.sample(range(A), 3)) #num_scheduled
                active_pos = anchor_positions[active_idx]    # [Kt, 2]
                d_active   = d_t[:, active_idx]              # [B, Kt]
                obs_t = build_observation_set(d_active, active_pos, B, device)
                step_out = dssm.step(z_post, h_gru, obs_t, active_pos)

            elif mode == 'scheduled' and actor is not None:
                # 先用先验预测，再选择锚点
                imagine_out = dssm.imagine(z_post, h_gru, anchor_positions)
                z_prior = imagine_out['z_prior_sample']
                h_t     = imagine_out['h_t']


                # 贪婪调度（评估时用 argmax）
                selected_idx, _, _ = actor.sample_action(z_prior, h_t, greedy=True)


                # 用真实测量（目标环境有真实数据用于评估）更新后验
                obs_list = []
                for b in range(B):
                    obs_b = []
                    for k_i in selected_idx[b]:
                        k = k_i.item()
                        d_k = d_t[b, k:k+1]              # [1]
                        p_k = anchor_positions[k]         # [2]
                        obs_b.append(torch.cat([d_k, p_k]))  # [3]
                    obs_list.append(torch.stack(obs_b, dim=0))  # [Kt, 3]
                obs_t = torch.stack(obs_list, dim=0)      # [B, Kt, 3]

                mu_post, std_post = dssm.encoder(z_post, h_t, obs_t)
                z_post_sample = reparameterize(mu_post, std_post)

                # 构造 step_out 格式
                step_out = {
                    'mu_post':      mu_post,
                    'std_post':     std_post,
                    'z_post_sample': z_post_sample,
                    'h_t':          h_t,
                    'h_gru':        dssm.sequence_model(z_post_sample, imagine_out['h_gru'])[1],
                }
            else:
                raise ValueError(f"未知评估模式：{mode}")

            # 提取位置估计（后验均值的位置分量）
            pos_est = dssm.get_position_estimate(step_out['mu_post'])  # [B, 2]

            all_pred.extend(pos_est.cpu().numpy())
            all_gt.extend(positions[:, t].cpu().numpy())

            # 更新状态
            z_post = step_out['z_post_sample'].detach()
            h_gru  = step_out['h_gru']

    return compute_all_metrics(
        np.array(all_pred), np.array(all_gt)
    )


# =========================================================================
# 主评估函数
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='LocDreamer 评估脚本（复现 Table 2）')
    parser.add_argument('--synthetic', action='store_true',
                        help='使用合成数据')
    parser.add_argument('--checkpoint', type=str, default='best',
                        help='检查点标签（best 或 epoch 编号）')
    parser.add_argument('--no-ekf', action='store_true',
                        help='跳过 EKF 基线（速度较慢）')
    parser.add_argument('--ekf-init', choices=['gt', 'ls', 'zero', 'center', 'random', 'all'],
                        default='gt',
                        help='EKF 初始位置：gt=真实首点，ls=第一帧测距最小二乘，zero=零初始化，center=房间中心，random=房间内随机，all=全部评估')
    parser.add_argument('--device', type=str, default=None,
                        help='覆盖计算设备，例如 cuda:2')
    parser.add_argument('--eval-method', '--eval-kind',
                        dest='eval_method',
                        choices=['stage1', 'dssm-all-real',
                                 'locdreamer-random', 'locdreamer-all',
                                 'locdreamer-scheduling'],
                        default='locdreamer-all',
                        help='评估方法：stage1、dssm-all-real、locdreamer-random、locdreamer-all、locdreamer-scheduling')
    parser.add_argument('--encoder-mode', choices=['mlp', 'wls'], default=None,
                        help='encoder 后验位置模式：mlp=论文原方法，wls=几何增强')
    parser.add_argument('--normalize-obs', action='store_true',
                        help='仅归一化 encoder 输入的距离/锚点坐标')
    parser.add_argument('--use-range-mask', action='store_true',
                        help='mask 物理不可能的 UWB range，并用 anchor 中位数替换输入')
    parser.add_argument('--posterior-state-mode', choices=['sample', 'mean'], default=None,
                        help='后验状态用于重建/RNN的方式')
    parser.add_argument('--dt', type=float, default=None,
                        help='覆盖匀速运动模型时间步长')
    parser.add_argument('--decoder-max-std', type=float, default=None,
                        help='可选：限制 decoder 测距标准差上限')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='覆盖检查点目录')
    parser.add_argument('--data-path', type=str, default=None,
                        help='覆盖预处理数据目录')
    parser.add_argument('--source-ids', type=str, default=None,
                        help='源 anchor 全局编号，例如 0,3,6')
    parser.add_argument('--target-ids', type=str, default=None,
                        help='目标 anchor 全局编号，例如 1,2,4,5,7；默认与 source 互补')
    args = parser.parse_args()

    cfg = Config()
    if args.device is not None:
        cfg.device = args.device
    if args.encoder_mode is not None:
        cfg.encoder_mode = args.encoder_mode
    if args.normalize_obs:
        cfg.normalize_obs = True
    if args.use_range_mask:
        cfg.use_range_mask = True
    if args.posterior_state_mode is not None:
        cfg.posterior_state_mode = args.posterior_state_mode
    if args.dt is not None:
        cfg.dt = args.dt
    if args.decoder_max_std is not None:
        cfg.decoder_max_std = args.decoder_max_std
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.source_ids is not None:
        cfg.source_anchor_ids = parse_anchor_ids(args.source_ids)
        cfg.num_source = len(cfg.source_anchor_ids)
        if args.target_ids is None:
            cfg.target_anchor_ids = tuple(i for i in range(8) if i not in cfg.source_anchor_ids)
            cfg.num_anchors = len(cfg.target_anchor_ids)
    if args.target_ids is not None:
        cfg.target_anchor_ids = parse_anchor_ids(args.target_ids)
        cfg.num_anchors = len(cfg.target_anchor_ids)
    device = torch.device(cfg.device)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── 加载数据 ──
    data_loaders = get_data_loaders(cfg, use_synthetic=args.synthetic)
    target_anchor_pos = data_loaders['target_anchor_pos'].to(device)
    target_anchor_pos_np = target_anchor_pos.cpu().numpy()

    # ── 初始化模型 ──
    dssm = DSSM(
        state_dim=cfg.state_dim,
        hidden_dim=cfg.hidden_dim,
        rnn_layers=cfg.rnn_layers,
        encoder_dim=cfg.encoder_dim,
        num_heads=cfg.num_heads,
        mlp_hidden=cfg.mlp_hidden,
        dt=cfg.dt,
        min_std=cfg.min_std,
        encoder_mode=cfg.encoder_mode,
        posterior_state_mode=cfg.posterior_state_mode,
        normalize_obs=cfg.normalize_obs,
        env_width=cfg.env_width,
        env_height=cfg.env_height,
        decoder_max_std=cfg.decoder_max_std,
    ).to(device)

    actor = Actor(
        state_dim=cfg.state_dim,
        hidden_dim=cfg.hidden_dim,
        num_anchors=cfg.num_anchors,
        num_scheduled=cfg.num_scheduled,
        mlp_hidden=cfg.mlp_hidden,
    ).to(device)

    critic = Critic(
        state_dim=cfg.state_dim,
        hidden_dim=cfg.hidden_dim,
        mlp_hidden=cfg.mlp_hidden,
    ).to(device)

    # 加载训练好的模型
    try:
        load_checkpoint(cfg, dssm, actor, critic, args.checkpoint, device)
    except FileNotFoundError as e:
        print(f"[警告] {e}")
        print("[警告] 将使用未训练的模型（仅用于测试代码流程）")

    target_loader = data_loaders['target_val_loader']

    print("\n" + "="*65)
    print("评估结果（对应论文 Table 2）")
    print("="*65)
    print(f"[配置] encoder_mode={cfg.encoder_mode}, "
          f"normalize_obs={cfg.normalize_obs}, "
          f"posterior_state_mode={cfg.posterior_state_mode}, "
          f"use_range_mask={cfg.use_range_mask}, "
          f"decoder_max_std={cfg.decoder_max_std}, checkpoint_dir={cfg.checkpoint_dir}")
    print(f"[配置] source_anchor_ids={cfg.source_anchor_ids}, "
          f"target_anchor_ids={cfg.target_anchor_ids}, dt={cfg.dt}")
    print(f"{'方法':<40} | MAE  | RMSE | 50th | 90th")
    print("-"*65)

    results = {}

    # ── 1. EKF 基线 ──
    if not args.no_ekf:
        print("[EKF] 正在评估...")

        ekf_init_modes = ['zero', 'center', 'random', 'ls', 'gt'] if args.ekf_init == 'all' else [args.ekf_init]
        for init_mode in ekf_init_modes:
            suffix = f" ({init_mode}-init)" if args.ekf_init == 'all' else ""

            m = evaluate_ekf(
                target_loader, target_anchor_pos_np,
                cfg.num_scheduled, all_anchors=False,
                init_mode=init_mode,
                env_width=cfg.env_width, env_height=cfg.env_height
            )
            results[f'EKF-random-{init_mode}'] = m
            print_metrics(m, f"EKF - random scheduling{suffix}")

            m = evaluate_ekf(
                target_loader, target_anchor_pos_np,
                all_anchors=True,
                init_mode=init_mode,
                env_width=cfg.env_width, env_height=cfg.env_height
            )
            results[f'EKF-all-{init_mode}'] = m
            print_metrics(m, f"EKF - all anchors{suffix}")

    # ── 2. DSSM / LocDreamer 变体 ──
    print("[DSSM] 正在评估...")

    if args.eval_method == 'dssm-all-real':
        m = evaluate_dssm(dssm, target_loader, target_anchor_pos, device,
                          mode='all')
        results['DSSM-all-real'] = m
        print_metrics(m, "DSSM - all anchors (real)")
    elif args.eval_method == 'stage1':
        m = evaluate_dssm(dssm, target_loader, target_anchor_pos, device,
                          mode='random', num_scheduled=cfg.num_scheduled)
        results['DSSM-stage1-random-target'] = m
        print_metrics(m, "DSSM Stage1 - random target anchors")

        m = evaluate_dssm(dssm, target_loader, target_anchor_pos, device,
                          mode='all')
        results['DSSM-stage1-all-target'] = m
        print_metrics(m, "DSSM Stage1 - all target anchors")
    elif args.eval_method == 'locdreamer-random':
        m = evaluate_dssm(dssm, target_loader, target_anchor_pos, device,
                          mode='random', num_scheduled=cfg.num_scheduled)
        results['LocDreamer-random'] = m
        print_metrics(m, "LocDreamer - random scheduling")
    elif args.eval_method == 'locdreamer-all':
        m = evaluate_dssm(dssm, target_loader, target_anchor_pos, device,
                          mode='all')
        results['LocDreamer-all-imagined'] = m
        print_metrics(m, "LocDreamer - all anchors (imagined)")
    elif args.eval_method == 'locdreamer-scheduling':
        m = evaluate_dssm(dssm, target_loader, target_anchor_pos, device,
                          actor=actor, mode='scheduled',
                          num_scheduled=cfg.num_scheduled)
        results['LocDreamer-scheduling'] = m
        print_metrics(m, "LocDreamer - scheduling  [主方法]")
    else:
        raise ValueError(f"未知评估方法：{args.eval_method}")

    print("="*65)
    print("\n论文参考值（Table 2）:")
    print(f"  EKF-random:           MAE=1.05, RMSE=1.67, 50th=0.71, 90th=1.83")
    print(f"  EKF-all:              MAE=0.92, RMSE=1.50, 50th=0.63, 90th=1.41")
    print(f"  DSSM-all(real):       MAE=0.57, RMSE=0.64, 50th=0.54, 90th=0.96")
    print(f"  LocDreamer-random:    MAE=1.07, RMSE=1.43, 50th=0.82, 90th=2.05")
    print(f"  LocDreamer-all(imag): MAE=0.85, RMSE=1.07, 50th=0.68, 90th=1.60")
    print(f"  LocDreamer-sched:     MAE=0.66, RMSE=0.77, 50th=0.60, 90th=1.18")

    return results


if __name__ == '__main__':
    main()
