"""
LocDreamer 训练脚本

对应论文 "Proposed LocDreamer Learning" 节及 Algorithm 1。

两阶段训练流程：
  ┌─────────────────────────────────────────────────────────────────┐
  │ Stage 1：DSSM 预训练（Algorithm 1, Lines 2-11）                  │
  │   - 输入：源环境的真实测量数据（3个锚点 R）                         │
  │   - 目标：最大化 ELBO，学习目标动态和环境特性                        │
  │   - 更新：DSSM 所有参数（θz, θh, θd, φ）                         │
  ├─────────────────────────────────────────────────────────────────┤
  │ Stage 2：WM 想象训练（Algorithm 1, Lines 12-24）                  │
  │   - 输入：无需真实数据！使用 DSSM 在新环境（5个锚点 A）中想象测量      │
  │   - 目标：同时微调 DSSM 并训练 RL 调度器（Actor-Critic）            │
  │   - 更新：DSSM 参数微调 + Actor（φ_α）+ Critic（φ_V）              │
  └─────────────────────────────────────────────────────────────────┘
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

from config import Config
from models.dssm import DSSM, reparameterize, kl_gaussian
from models.actor_critic import (
    Actor, Critic,
    compute_returns, compute_actor_loss, compute_critic_loss
)
from utils.dataset import get_data_loaders
from utils.metrics import compute_all_metrics, print_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def parse_anchor_ids(text: str) -> Tuple[int, ...]:
    """Parse comma-separated anchor ids, e.g. '0,3,6'."""
    ids = tuple(int(x.strip()) for x in text.split(',') if x.strip())
    if len(set(ids)) != len(ids):
        raise ValueError(f"anchor ids 存在重复: {text}")
    return ids


def create_summary_writer(cfg: Config, args):
    """Create a TensorBoard writer and record the run config."""
    if getattr(args, 'no_tensorboard', False):
        return None
    if SummaryWriter is None:
        print("[TensorBoard] tensorboard is not installed; skip event logging.")
        return None

    log_dir = args.tensorboard_dir
    if log_dir is None:
        log_dir = os.path.join(cfg.checkpoint_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=log_dir)
    config_lines = [
        f"experiment={args.experiment}",
        f"source_anchor_ids={cfg.source_anchor_ids}",
        f"target_anchor_ids={cfg.target_anchor_ids}",
        f"seq_len={cfg.seq_len}",
        f"dssm_epochs={cfg.dssm_epochs}",
        f"imagine_epochs={cfg.imagine_epochs}",
        f"stage2_policy={cfg.stage2_policy}",
        f"encoder_mode={cfg.encoder_mode}",
        f"posterior_state_mode={cfg.posterior_state_mode}",
        f"decoder_max_std={cfg.decoder_max_std}",
        f"checkpoint_dir={cfg.checkpoint_dir}",
        f"data_path={cfg.data_path}",
    ]
    writer.add_text('config/text_summary', '\n'.join(config_lines), 0)
    print(f"[TensorBoard] logging to {log_dir}")
    return writer


def log_scalar_dict(writer, prefix: str, metrics: Dict[str, float], step: int):
    if writer is None:
        return
    for name, value in metrics.items():
        if value is not None:
            writer.add_scalar(f"{prefix}/{name}", float(value), step)


def log_optimizer_lr(writer, prefix: str, optimizer: optim.Optimizer, step: int):
    if writer is None:
        return
    for group_idx, group in enumerate(optimizer.param_groups):
        suffix = '' if group_idx == 0 else f'_{group_idx}'
        writer.add_scalar(f"{prefix}/lr{suffix}", float(group['lr']), step)


def capture_rng_state() -> Dict:
    """Capture RNG states so validation does not perturb training randomness."""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': None,
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


@torch.no_grad()
def evaluate_dssm_for_tensorboard(dssm: DSSM,
                                  data_loader,
                                  anchor_positions: torch.Tensor,
                                  device: torch.device,
                                  actor: Actor = None,
                                  num_scheduled: int = 3,
                                  mode: str = 'all') -> Dict[str, float]:
    """Evaluate DSSM/LocDreamer on a validation loader without EKF baselines."""
    was_training = dssm.training
    actor_was_training = actor.training if actor is not None else False
    rng_state = capture_rng_state()

    try:
        dssm.eval()
        if actor is not None:
            actor.eval()

        num_anchors = anchor_positions.shape[0]
        all_pred = []
        all_gt = []

        for batch in data_loader:
            distances = batch['distances'].to(device)
            positions = batch['positions'].to(device)
            B, T, _ = distances.shape
            z_post, h_gru = dssm.init_state(B, device)

            for t in range(T):
                d_t = distances[:, t, :]

                if mode == 'all':
                    obs_t = build_observation_set(d_t, anchor_positions, B, device)
                    step_out = dssm.step(z_post, h_gru, obs_t, anchor_positions)

                elif mode == 'random':
                    active_idx = sorted(random.sample(range(num_anchors), num_scheduled))
                    active_pos = anchor_positions[active_idx]
                    d_active = d_t[:, active_idx]
                    obs_t = build_observation_set(d_active, active_pos, B, device)
                    step_out = dssm.step(z_post, h_gru, obs_t, active_pos)

                elif mode == 'scheduled' and actor is not None:
                    imagine_out = dssm.imagine(z_post, h_gru, anchor_positions)
                    selected_idx, _, _ = actor.sample_action(
                        imagine_out['z_prior_sample'], imagine_out['h_t'], greedy=True
                    )
                    obs_list = []
                    for b in range(B):
                        obs_b = []
                        for k_i in selected_idx[b]:
                            k = k_i.item()
                            obs_b.append(torch.cat([d_t[b, k:k + 1], anchor_positions[k]]))
                        obs_list.append(torch.stack(obs_b, dim=0))
                    obs_t = torch.stack(obs_list, dim=0)

                    mu_post, std_post = dssm.encoder(z_post, imagine_out['h_t'], obs_t)
                    z_post_sample = reparameterize(mu_post, std_post)
                    step_out = {
                        'mu_post': mu_post,
                        'std_post': std_post,
                        'z_post_sample': z_post_sample,
                        'h_t': imagine_out['h_t'],
                        'h_gru': dssm.sequence_model(z_post_sample, imagine_out['h_gru'])[1],
                    }
                else:
                    raise ValueError(f"unknown evaluation mode: {mode}")

                pos_est = dssm.get_position_estimate(step_out['mu_post'])
                all_pred.extend(pos_est.cpu().numpy())
                all_gt.extend(positions[:, t].cpu().numpy())
                z_post = step_out['z_post_sample'].detach()
                h_gru = step_out['h_gru']

        return compute_all_metrics(np.array(all_pred), np.array(all_gt))
    finally:
        restore_rng_state(rng_state)
        if was_training:
            dssm.train()
        if actor is not None and actor_was_training:
            actor.train()


# =========================================================================
# Stage 1：DSSM 预训练
# =========================================================================

def train_dssm_epoch(dssm: DSSM,
                     optimizer: optim.Optimizer,
                     data_loader,
                     source_anchor_pos: torch.Tensor,
                     device: torch.device,
                     pos_loss_weight: float = 0.0,
                     detach_state_between_steps: bool = True) -> Dict[str, float]:
    """
    单轮 DSSM 预训练（Algorithm 1, Lines 3-11 的批量版本）

    对每个序列批次：
      1. 初始化 z_0^post 和 GRU 隐藏状态
      2. 时序展开：预测先验→推断后验→重建距离→更新 RNN
      3. 累积 ELBO 损失，反向传播更新参数

    Args:
        dssm:              DSSM 模型
        optimizer:         Adam 优化器
        data_loader:       源环境数据加载器
        source_anchor_pos: [|R|, 2]，源锚点位置（固定，已在 device 上）
        device:            计算设备

    Returns:
        包含平均损失的字典
    """
    dssm.train()
    total_loss = total_recon = total_dyn = 0.0
    num_batches = 0

    for batch in data_loader:
        # batch['distances']: [B, T, |R|]
        # batch['positions']: [B, T, 2]（仅用于评估，不用于训练）
        distances = batch['distances'].to(device)   # [B, T, |R|]
        positions = batch['positions'].to(device)   # [B, T, 2]
        distance_masks = batch.get('distance_masks')
        if distance_masks is not None:
            distance_masks = distance_masks.to(device)
        B, T, num_source = distances.shape

        # ── 初始化（Algorithm 1, Line 3）──
        z_post, h_gru = dssm.init_state(B, device)

        # ── 时序展开（Algorithm 1, Lines 4-10）──
        total_elbo = torch.tensor(0.0, device=device)
        total_pos_loss_seq = torch.tensor(0.0, device=device)

        for t in range(T):
            # 当前时间步的距离观测 [B, |R|]
            d_t = distances[:, t, :]
            mask_t = distance_masks[:, t, :] if distance_masks is not None else None

            # 构造观测集合 o(α_t)：{(d_t^r, p^r) | r ∈ R}，形状 [B, |R|, 3]
            # 每个元素为 (distance, anchor_x, anchor_y)
            obs_t = build_observation_set(d_t, source_anchor_pos, B, device)

            # ── 前向传播（式 7, 9, 11, 13）──
            step_out = dssm.step(z_post, h_gru, obs_t, source_anchor_pos)

            # ── 计算 ELBO 损失（式 6）──
            l_recon, l_dyn = dssm.compute_elbo(
                d_obs=d_t,
                mu_prior=step_out['mu_prior'],
                std_prior=step_out['std_prior'],
                mu_post=step_out['mu_post'],
                std_post=step_out['std_post'],
                z_post_sample=step_out['z_post_sample'],
                h_t=step_out['h_t'],
                anchor_positions=source_anchor_pos,
                obs_mask=mask_t,
            )

            # ELBO = L_recon - L_dyn（最大化 ELBO，最小化 -ELBO）
            elbo_t = l_recon - l_dyn
            total_elbo = total_elbo + elbo_t
            total_recon += l_recon.item()
            total_dyn   += l_dyn.item()
            if pos_loss_weight > 0:
                total_pos_loss_seq = total_pos_loss_seq + F.mse_loss(
                    step_out['mu_post'][..., :2], positions[:, t, :]
                )

            # ── 更新 RNN 隐藏状态 h_{t+1}（式 7）──
            # 注意：用后验状态更新 RNN，以便后续步骤利用已编码的信息
            if detach_state_between_steps:
                z_post = step_out['z_post_sample'].detach()
            else:
                z_post = step_out['z_post_sample']
            h_gru  = step_out['h_gru']

        # ── 反向传播（对序列取平均）──
        loss = -total_elbo / T   # 最小化 -ELBO_mean
        if pos_loss_weight > 0:
            loss = loss + pos_loss_weight * total_pos_loss_seq / T
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止 RNN 梯度爆炸
        nn.utils.clip_grad_norm_(dssm.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {
        'loss':  total_loss / max(num_batches, 1),
        'recon': total_recon / max(num_batches * T, 1),
        'dyn':   total_dyn   / max(num_batches * T, 1),
    }


def pretrain_dssm(cfg: Config, dssm: DSSM,
                  data_loaders: Dict,
                  device: torch.device,
                  loader_key: str = 'source_train_loader',
                  anchor_key: str = 'source_anchor_pos',
                  checkpoint_name: str = 'dssm_pretrained.pt',
                  title: str = 'DSSM 预训练（使用源环境真实数据）',
                  writer=None,
                  tb_prefix: str = 'stage1',
                  validate_source: bool = True,
                  validate_target_random: bool = True) -> DSSM:
    """
    Stage 1：DSSM 预训练主循环（Algorithm 1, Lines 2-11）

    在源环境真实测量上训练 DSSM，使其学习目标动态特性。
    训练完成后，DSSM 可以想象任意锚点配置下的测量。
    """
    print("\n" + "="*60)
    print(f"Stage 1: {title}")
    print("="*60)
 
    train_anchor_pos = data_loaders[anchor_key].to(device)

    # Adam 优化器（Table 1：lr=1e-3，weight_decay=1e-3）
    optimizer = optim.Adam(
        dssm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # Cosine Annealing 学习率调度（Table 1）
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.dssm_epochs, eta_min=1e-5)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, cfg.dssm_epochs + 1):
        # 训练
        train_metrics = train_dssm_epoch(
            dssm, optimizer, data_loaders[loader_key],
            train_anchor_pos, device, cfg.pos_loss_weight,
            cfg.detach_state_between_steps
        )
        scheduler.step()

        log_scalar_dict(writer, f'{tb_prefix}/train', {
            'loss': train_metrics['loss'],
            'recon': train_metrics['recon'],
            'kl': train_metrics['dyn'],
        }, epoch)
        if writer is not None:
            writer.add_scalar(f'{tb_prefix}/lr/dssm', optimizer.param_groups[0]['lr'], epoch)

        if epoch % cfg.log_interval == 0:
            print(
                f"[预训练] Epoch {epoch:3d}/{cfg.dssm_epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Recon: {train_metrics['recon']:.4f} | "
                f"KL: {train_metrics['dyn']:.4f}"
            )

        tb_eval_interval = getattr(cfg, 'tensorboard_eval_interval', cfg.log_interval)
        if writer is not None and epoch % tb_eval_interval == 0:
            if validate_source:
                source_metrics = evaluate_dssm_for_tensorboard(
                    dssm, data_loaders['source_val_loader'],
                    data_loaders['source_anchor_pos'].to(device),
                    device, mode='all'
                )
                log_scalar_dict(writer, f'{tb_prefix}/val_all_source', source_metrics, epoch)

            target_anchor_pos = data_loaders['target_anchor_pos'].to(device)
            target_all_metrics = evaluate_dssm_for_tensorboard(
                dssm, data_loaders['target_val_loader'],
                target_anchor_pos, device, mode='all'
            )
            log_scalar_dict(writer, f'{tb_prefix}/val_all_target', target_all_metrics, epoch)

            if validate_target_random:
                target_random_metrics = evaluate_dssm_for_tensorboard(
                    dssm, data_loaders['target_val_loader'],
                    target_anchor_pos, device,
                    num_scheduled=cfg.num_scheduled, mode='random'
                )
                log_scalar_dict(writer, f'{tb_prefix}/val_random_target', target_random_metrics, epoch)

        # 保存最优检查点
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            torch.save(
                dssm.state_dict(),
                os.path.join(cfg.checkpoint_dir, checkpoint_name)
            )
        
        # 每隔 50 epoch 保存一份“截至当前 epoch 的最优模型”
        if epoch % 50 == 0:
            best_state = torch.load(
                os.path.join(cfg.checkpoint_dir, checkpoint_name),
                map_location=device
            )
            torch.save(
                best_state,
                os.path.join(cfg.checkpoint_dir, f'dssm_best_until_{epoch}.pt')
            )
            # torch.save(
            #     dssm.state_dict(),
            #     os.path.join(cfg.checkpoint_dir, f'dssm_epoch_{epoch}.pt')
            # )
            

    print(f"\n[预训练] 完成。最优损失: {best_loss:.4f}")
    # 加载最优权重
    dssm.load_state_dict(
        torch.load(os.path.join(cfg.checkpoint_dir, checkpoint_name),
                   map_location=device)
    )
    return dssm


# =========================================================================
# Stage 2：LocDreamer 想象训练（DSSM 微调 + AC 训练）
# =========================================================================

def imagination_epoch(dssm: DSSM, actor: Actor, critic: Critic,
                      dssm_optimizer: optim.Optimizer,
                      ac_optimizer: optim.Optimizer,
                      target_anchor_pos: torch.Tensor,
                      source_anchor_pos: torch.Tensor,
                      source_sequence_batch: Dict[str, torch.Tensor],
                      batch_size: int, seq_len: int,
                      gamma: float, device: torch.device,
                      pos_loss_weight: float = 0.0,
                      scheduler_pos_reward_weight: float = 0.0,
                      stage2_policy: str = "actor"
                      ) -> Dict[str, float]:
    """
    单轮想象训练（Algorithm 1, Lines 13-24）

    全流程在 DSSM "梦境"中完成，无需真实测量：
      1. 初始化状态
      2. 先验预测并生成想象测量（Algorithm 1, Lines 15-16）
      3. Actor 选择锚点（Line 17）
      4. 更新后验（Line 18）
      5. 重建距离并计算损失/奖励（Lines 19-23）
      6. 反向传播更新 DSSM（微调）和 AC（训练）（Line 24）

    Args:
        dssm, actor, critic: 模型
        dssm_optimizer:      DSSM 微调优化器
        ac_optimizer:        Actor-Critic 优化器
        target_anchor_pos:   [A, 2]，目标环境锚点位置
        source_anchor_pos:   [|R|, 2]，源环境锚点位置；想象阶段按论文
                             Algorithm 1 Line 19/22 对 R ∪ Kt 重建
        batch_size, seq_len: 批大小 B 和序列长度 T
        gamma:               折扣因子
        device:              计算设备

    Returns:
        各损失项的均值字典
    """
    dssm.train()
    actor.train()
    critic.train()

    A = target_anchor_pos.shape[0]   # 目标环境锚点总数（A=5）
    R = source_anchor_pos.shape[0]   # 源环境锚点数（|R|=3）
    union_anchor_pos = torch.cat([source_anchor_pos, target_anchor_pos], dim=0)
    source_distances = source_sequence_batch['distances'].to(device)
    source_positions = source_sequence_batch['positions'].to(device)
    if source_distances.shape[0] != batch_size:
        batch_size = source_distances.shape[0]

    # ── Algorithm 1, Line 13：初始化 ──
    z_post, h_gru = dssm.init_state(batch_size, device)

    # 用于收集轨迹数据（REINFORCE 需要完整轨迹）
    log_probs_list = []    # log π(α_t | s_t)，每步一个 [B]
    rewards_list   = []    # R_t = L_recon_t，每步一个 [B]（标量）
    values_list    = []    # V(s_t)，每步一个 [B]
    prior_states   = []    # (z_prior, h_t)，用于 Critic

    total_dssm_loss = torch.tensor(0.0, device=device)

    # ── Algorithm 1, Lines 14-23：时序展开 ──
    for t in range(seq_len):

        # ── Line 15：先验预测，同时生成想象测量（Line 16）──
        imagine_out = dssm.imagine(z_post, h_gru, union_anchor_pos)
        # imagine_out 包含：mu_prior, std_prior, z_prior_sample, h_t, h_gru,
        #                    imagined_distances [B, |R|+A], imagined_stds [B, |R|+A]
        z_prior_sample     = imagine_out['z_prior_sample']       # [B, Z]
        h_t                = imagine_out['h_t']                  # [B, H]
        imagined_distances = imagine_out['imagined_distances']   # [B, |R|+A]

        # ── Line 17：选择锚点 α_t ──
        # actor: 论文主方法；random/all: Table 2 的想象训练 baseline。
        if stage2_policy == "actor":
            # 对 DSSM 输出 detach，避免 dssm_optimizer.step() 原地更新参数后
            # 破坏 actor log_prob 的计算图（policy gradient 只需 actor 梯度）
            selected_idx, combo_idx, log_prob = actor.sample_action(
                z_prior_sample.detach(), h_t.detach(), greedy=False
            )
        elif stage2_policy == "random":
            rand = torch.stack([
                torch.randperm(A, device=device)[:actor.num_scheduled] #actor.num_scheduled
                for _ in range(batch_size)
            ], dim=0)
            selected_idx = rand.sort(dim=1).values
            log_prob = None
        # elif stage2_policy == "random":
        #     rand = torch.randperm(A, device=device)[:5] # actor.num_scheduled
        #     selected_idx = rand.sort().values.unsqueeze(0).expand(batch_size, -1)
        #     log_prob = None
        elif stage2_policy == "all":
            selected_idx = torch.arange(A, device=device).unsqueeze(0).expand(batch_size, -1)
            log_prob = None
        else:
            raise ValueError(f"未知 stage2_policy: {stage2_policy}")
        # selected_idx: [B, Kt]，选中新环境 A 内锚点的索引
        # log_prob:     [B]，log π(α_t | s_t)
        selected_union_idx = selected_idx + R

        # ── Line 18：后验推断 z_t^posterior（使用选中锚点的想象测量）──
        post_out = dssm.update_posterior(
            z_post, h_t, h_gru,
            imagined_distances, selected_union_idx, union_anchor_pos
        )
        z_post_sample = post_out['z_post_sample']   # [B, Z]
        h_gru = post_out['h_gru']

        # ── Lines 19-22：重建距离并计算 ELBO（奖励）──
        # 按论文为 R ∪ Kt 的锚点重建距离
        source_idx = torch.arange(R, device=device).unsqueeze(0).expand(batch_size, -1)
        recon_idx = torch.cat([source_idx, selected_union_idx], dim=1)
        l_recon_vec, l_dyn_vec = compute_imagine_elbo(
            dssm, imagine_out, post_out,
            recon_idx, union_anchor_pos, h_t
        )
        l_recon_t = l_recon_vec.mean()
        l_dyn_t = l_dyn_vec.mean()
        if pos_loss_weight > 0:
            d_source_t = source_distances[:, t, :]
            obs_source_t = build_observation_set(d_source_t, source_anchor_pos, batch_size, device)
            source_post_mu, _ = dssm.encoder(z_post, h_t, obs_source_t)
            pos_loss_t = F.mse_loss(source_post_mu[..., :2], source_positions[:, t, :])
        else:
            pos_loss_t = torch.tensor(0.0, device=device)

        # DSSM 微调损失（ELBO）
        total_dssm_loss = total_dssm_loss + (l_recon_t - l_dyn_t - pos_loss_weight * pos_loss_t)

        # ── Line 23：RL 奖励 R_t = L_recon_t（式 15, 16）──
        # 奖励是重建质量：锚点越有信息量，重建越好，奖励越高
        if scheduler_pos_reward_weight > 0:
            selected_wls_pos = dssm.encoder.wls_position(post_out['obs'])
            scheduler_pos_err = (selected_wls_pos - z_prior_sample.detach()[..., :2]).pow(2).mean(dim=-1)
            reward = (l_recon_vec - scheduler_pos_reward_weight * scheduler_pos_err).detach()
        else:
            reward = l_recon_vec.detach()  # [B]
        rewards_list.append(reward)

        # Critic 估计当前状态价值
        # 用先验状态 s_t = (z_prior, h_t) 作为 Critic 输入
        value = critic(z_prior_sample.detach(), h_t.detach()).squeeze(-1)  # [B]
        values_list.append(value)
        if log_prob is not None:
            log_probs_list.append(log_prob)   # [B]

        # 记录先验状态（用于后续 Critic 梯度更新）
        prior_states.append((z_prior_sample.detach(), h_t.detach()))

        # 更新后验状态（截断梯度，防止无限 BPTT）
        z_post = z_post_sample.detach()

    # ── Line 24：更新 DSSM（微调）──
    dssm_loss = -total_dssm_loss / seq_len
    dssm_optimizer.zero_grad()
    dssm_loss.backward(retain_graph=True)
    nn.utils.clip_grad_norm_(dssm.parameters(), max_norm=10.0)
    dssm_optimizer.step()

    # ── Line 24：更新 Actor 和 Critic（REINFORCE 算法）──
    if stage2_policy == "actor":
        ac_loss_dict = update_actor_critic(
            actor, critic, ac_optimizer,
            log_probs_list, rewards_list, values_list,
            prior_states, gamma, device
        )
    else:
        ac_loss_dict = {'actor_loss': 0.0, 'critic_loss': 0.0}

    mean_reward = torch.stack(rewards_list, dim=0).mean()

    return {
        'dssm_loss': dssm_loss.item(),
        'mean_reward': mean_reward.item(),
        **ac_loss_dict,
    }


def compute_imagine_elbo(dssm: DSSM,
                         imagine_out: Dict, post_out: Dict,
                         recon_idx: torch.Tensor,
                         all_anchor_pos: torch.Tensor,
                         h_t: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算想象阶段的 ELBO（Algorithm 1, Lines 21-22）

    想象距离 d_t^k（来自先验生成）作为"真实"距离，
    后验重建 d̂_t^k 与其对比计算重建损失。

    Args:
        dssm:         DSSM 模型
        imagine_out:  imagine() 的输出字典
        post_out:     update_posterior() 的输出字典
        recon_idx:    [B, Kr]，需要重建的锚点索引，论文想象阶段为 R ∪ Kt
        all_anchor_pos: [|R|+A, 2]，参与想象的所有锚点位置
        h_t:          [B, H]，RNN 隐藏状态

    Returns:
        l_recon: 重建对数概率（标量）
        l_dyn:   KL 散度（标量）
    """
    B, Kr = recon_idx.shape
    device = h_t.device

    # ── 重建损失：后验状态重建选中锚点的距离 ──
    # 想象"真实"距离（来自先验的 reparameterize）
    imagined_d = imagine_out['imagined_distances']   # [B, |R|+A]

    z_post_sample = post_out['z_post_sample']        # [B, Z]

    l_recon_total = torch.zeros(B, device=device)
    for b in range(B):
        for i in range(Kr):
            k = recon_idx[b, i].item()
            p_k = all_anchor_pos[k].unsqueeze(0)                    # [1, 2]
            z_b = z_post_sample[b:b+1]                              # [1, Z]
            h_b = h_t[b:b+1]                                        # [1, H]
            d_imag_bk = imagined_d[b:b+1, k:k+1]                   # [1, 1]

            mu_d, std_d = dssm.decoder(z_b, h_b, p_k)              # [1, 1]
            log_p = dssm.decoder.log_prob(d_imag_bk, mu_d, std_d)  # [1, 1]
            l_recon_total[b] = l_recon_total[b] + log_p.squeeze()

    l_recon = l_recon_total / Kr

    # ── KL 散度：后验 vs 先验 ──
    kl = kl_gaussian(
        post_out['mu_post'], post_out['std_post'],
        imagine_out['mu_prior'], imagine_out['std_prior']
    )   # [B, Z]
    l_dyn = kl.sum(dim=-1)

    return l_recon, l_dyn


def update_actor_critic(actor: Actor, critic: Critic,
                        optimizer: optim.Optimizer,
                        log_probs_list: List,
                        rewards_list: List,
                        values_list: List,
                        prior_states: List,
                        gamma: float,
                        device: torch.device) -> Dict[str, float]:
    """
    用收集到的轨迹数据更新 Actor 和 Critic（REINFORCE with baseline）

    对应论文式 (16) 和 (17)。

    步骤：
      1. 计算折扣回报 G_t = Σ_{τ≥t} γ^{τ-t} R_τ
      2. 计算优势函数 Adv_t = G_t - V(s_t)（基线降低方差）
      3. Actor 损失 = -E[log π(α_t|s_t) * Adv_t]
      4. Critic 损失 = E[(V(s_t) - G_t)²]
    """
    T = len(rewards_list)
    B = log_probs_list[0].shape[0]

    # ── 计算逐样本折扣回报 G_t ──
    # rewards_list 中每个元素为 [B]。不要先做 batch mean，否则同一 batch
    # 内不同 anchor 组合会拿到同一个 reward，Actor 学不到组合差异。
    rewards_tensor = torch.stack(rewards_list, dim=0).to(device)  # [T, B]
    returns = torch.zeros_like(rewards_tensor)
    G = torch.zeros(B, device=device)
    for t in reversed(range(T)):
        G = rewards_tensor[t] + gamma * G
        returns[t] = G
    # 归一化回报（减少训练方差）
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # ── 计算优势函数 Adv_t = G_t - V(s_t)（batch 维度的平均值） ──
    log_probs_tensor = torch.stack(log_probs_list, dim=0)   # [T, B]
    values_tensor    = torch.stack(values_list, dim=0)      # [T, B]
    # 优势 = 实际回报 - 基线（Critic 估计）
    advantages = returns - values_tensor.detach()  # [T, B]

    # ── Actor 损失（式 16）──
    actor_loss = compute_actor_loss(log_probs_tensor, advantages)

    # ── Critic 损失（式 17）──
    critic_loss = compute_critic_loss(values_tensor, returns)

    # 联合更新 Actor 和 Critic
    total_ac_loss = actor_loss + 0.5 * critic_loss
    optimizer.zero_grad()
    total_ac_loss.backward()
    nn.utils.clip_grad_norm_(
        list(actor.parameters()) + list(critic.parameters()), max_norm=10.0
    )
    optimizer.step()

    return {
        'actor_loss':  actor_loss.item(),
        'critic_loss': critic_loss.item(),
    }


def train_locdreamer(cfg: Config, dssm: DSSM, actor: Actor, critic: Critic,
                     data_loaders: Dict,
                     device: torch.device,
                     writer=None):
    """
    Stage 2：LocDreamer 想象训练主循环（Algorithm 1, Lines 12-24）

    不需要目标环境的真实数据，完全基于 DSSM 的想象能力训练调度策略。
    """
    print("\n" + "="*60)
    print("Stage 2: LocDreamer 想象训练（无需真实数据）")
    print("="*60)

    target_anchor_pos = data_loaders['target_anchor_pos'].to(device)
    source_anchor_pos = data_loaders['source_anchor_pos'].to(device)

    # DSSM 微调优化器（较小学习率，避免破坏预训练结果）
    dssm_optimizer = optim.Adam(
        dssm.parameters(), lr=cfg.lr * 0.1, weight_decay=cfg.weight_decay
    )
    # Actor-Critic 优化器
    ac_optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # Cosine Annealing 调度（Table 1）
    dssm_scheduler = CosineAnnealingLR(dssm_optimizer, T_max=cfg.imagine_epochs)
    ac_scheduler   = CosineAnnealingLR(ac_optimizer,   T_max=cfg.imagine_epochs)

    best_stage2_metric = float('inf')
    source_iter = iter(data_loaders['source_train_loader'])

    for epoch in range(1, cfg.imagine_epochs + 1):
        try:
            source_sequence_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(data_loaders['source_train_loader'])
            source_sequence_batch = next(source_iter)
        metrics = imagination_epoch(
            dssm, actor, critic,
            dssm_optimizer, ac_optimizer,
            target_anchor_pos,
            source_anchor_pos,
            source_sequence_batch,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            gamma=cfg.gamma,
            device=device,
            pos_loss_weight=cfg.pos_loss_weight,
            scheduler_pos_reward_weight=cfg.scheduler_pos_reward_weight,
            stage2_policy=cfg.stage2_policy,
        )
        dssm_scheduler.step()
        if cfg.stage2_policy == "actor":
            ac_scheduler.step()

        log_scalar_dict(writer, 'stage2/train', {
            'dssm_loss': metrics['dssm_loss'],
            'reward': metrics['mean_reward'],
            'actor_loss': metrics['actor_loss'],
            'critic_loss': metrics['critic_loss'],
        }, epoch)
        if writer is not None:
            writer.add_scalar('stage2/lr/dssm', dssm_optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('stage2/lr/ac', ac_optimizer.param_groups[0]['lr'], epoch)

        if epoch % cfg.log_interval == 0:
            print(
                f"[想象训练] Epoch {epoch:4d}/{cfg.imagine_epochs} | "
                f"DSSM: {metrics['dssm_loss']:.4f} | "
                f"Reward: {metrics['mean_reward']:.4f} | "
                f"Actor: {metrics['actor_loss']:.4f} | "
                f"Critic: {metrics['critic_loss']:.4f}"
            )

        tb_eval_interval = getattr(cfg, 'tensorboard_eval_interval', cfg.log_interval)
        if writer is not None and epoch % tb_eval_interval == 0:
            if cfg.stage2_policy == "actor":
                target_scheduled_metrics = evaluate_dssm_for_tensorboard(
                    dssm, data_loaders['target_val_loader'],
                    target_anchor_pos, device, actor=actor,
                    num_scheduled=cfg.num_scheduled, mode='scheduled'
                )
                log_scalar_dict(writer, 'stage2/val_scheduling', target_scheduled_metrics, epoch)
            elif cfg.stage2_policy == "all":
                target_all_metrics = evaluate_dssm_for_tensorboard(
                    dssm, data_loaders['target_val_loader'],
                    target_anchor_pos, device, mode='all'
                )
                log_scalar_dict(writer, 'stage2/val_all_target', target_all_metrics, epoch)

                target_random_metrics = evaluate_dssm_for_tensorboard(
                    dssm, data_loaders['target_val_loader'],
                    target_anchor_pos, device,
                    num_scheduled=cfg.num_scheduled, mode='random'
                )
                log_scalar_dict(writer, 'stage2/val_random_target', target_random_metrics, epoch)
            elif cfg.stage2_policy == "random":
                target_random_metrics = evaluate_dssm_for_tensorboard(
                    dssm, data_loaders['target_val_loader'],
                    target_anchor_pos, device,
                    num_scheduled=cfg.num_scheduled, mode='random'
                )
                log_scalar_dict(writer, 'stage2/val_random_target', target_random_metrics, epoch)

        # 保存截至当前 epoch 的最优 Stage 2 检查点。
        # actor 策略使用 mean_reward；random/all 没有 Actor 更新，用 DSSM loss 更有意义。
        monitor_metric = (
            -metrics['mean_reward'] # metrics['actor_loss'] metrics['dssm_loss']
            if cfg.stage2_policy == "actor"
            else metrics['dssm_loss']
        )
        if monitor_metric < best_stage2_metric:
            best_stage2_metric = monitor_metric
            save_checkpoint(cfg, dssm, actor, critic, 'best')

        # 每 save_interval 轮保存当前检查点，以及“截至当前 epoch 的最优检查点”
        if epoch % cfg.save_interval == 0: # cfg.save_interval
            save_checkpoint(cfg, dssm, actor, critic, epoch)

            best_path = os.path.join(cfg.checkpoint_dir, 'locdreamer_best.pt')
            if os.path.exists(best_path):
                best_state = torch.load(best_path, map_location=device)
                torch.save(
                    best_state,
                    os.path.join(
                        cfg.checkpoint_dir,
                        f'locdreamer_best_until_{epoch}.pt'
                    )
                )

    print(f"\n[想象训练] 完成。")


# =========================================================================
# 辅助函数
# =========================================================================

def build_observation_set(d_t: torch.Tensor,
                           anchor_positions: torch.Tensor,
                           B: int, device: torch.device) -> torch.Tensor:
    """
    构造 Set Transformer 的输入：观测集合 o(α_t) = {(d_t^k, p^k)}

    将距离测量和锚点位置拼接为集合格式。

    Args:
        d_t:             [B, K]  — 当前步各锚点的距离测量
        anchor_positions: [K, 2] — 锚点位置
        B:               批大小
        device:          计算设备

    Returns:
        obs: [B, K, 3]  — 每个元素为 [distance, anchor_x, anchor_y]
    """
    K = anchor_positions.shape[0]
    # 将锚点位置广播到 batch 维度：[K, 2] → [B, K, 2]
    pos_expanded = anchor_positions.unsqueeze(0).expand(B, -1, -1)   # [B, K, 2]
    # 将距离增加维度：[B, K] → [B, K, 1]
    d_expanded = d_t.unsqueeze(-1)                                    # [B, K, 1]
    # 拼接：[B, K, 1] + [B, K, 2] → [B, K, 3]
    obs = torch.cat([d_expanded, pos_expanded], dim=-1)               # [B, K, 3]
    return obs


def save_checkpoint(cfg: Config, dssm: DSSM, actor: Actor, critic: Critic,
                    tag: str = ''):
    """保存模型检查点"""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    torch.save({
        'dssm':   dssm.state_dict(),
        'actor':  actor.state_dict(),
        'critic': critic.state_dict(),
    }, os.path.join(cfg.checkpoint_dir, f'locdreamer_{tag}.pt'))


def load_checkpoint(cfg: Config, dssm: DSSM, actor: Actor, critic: Critic,
                    tag: str = 'best', device: torch.device = None):
    """加载模型检查点"""
    # path = os.path.join(cfg.checkpoint_dir, f'locdreamer_{tag}.pt')
    if tag.endswith(".pt") or os.path.exists(tag):
        path = tag
    else:
        path = os.path.join(cfg.checkpoint_dir, f"locdreamer_{tag}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"检查点不存在：{path}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'dssm' in ckpt:
        dssm.load_state_dict(ckpt['dssm'])
        if 'actor' in ckpt:
            actor.load_state_dict(ckpt['actor'])
        if 'critic' in ckpt:
            critic.load_state_dict(ckpt['critic'])
        print(f"[检查点] 已加载完整 LocDreamer checkpoint：{path}")
    else:
        dssm.load_state_dict(ckpt)
        print(f"[检查点] 已加载 DSSM-only checkpoint：{path}")


# =========================================================================
# 主函数
# =========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LocDreamer 训练脚本')
    parser.add_argument('--experiment',
                        choices=['locdreamer', 'dssm-all-real'],
                        default='locdreamer',
                        help='训练实验：locdreamer=两阶段想象训练；dssm-all-real=目标5锚点真实数据DSSM baseline')
    parser.add_argument('--synthetic', action='store_true',
                        help='使用合成数据（无需下载真实数据集）')
    parser.add_argument('--dssm-only', action='store_true',
                        help='仅运行 Stage 1 预训练')
    parser.add_argument('--dssm-epochs', type=int, default=None,
                        help='覆盖 DSSM 预训练轮数')
    parser.add_argument('--imagine-epochs', type=int, default=None,
                        help='覆盖想象训练轮数（论文用 1000 测试过拟合）')
    parser.add_argument('--device', type=str, default=None,
                        help='覆盖计算设备，例如 cuda:2')
    parser.add_argument('--pos-loss-weight', type=float, default=None,
                        help='覆盖位置辅助损失权重')
    parser.add_argument('--scheduler-pos-reward-weight', type=float, default=None,
                        help='覆盖调度器几何定位一致性奖励权重')
    parser.add_argument('--encoder-mode', choices=['mlp', 'wls'], default=None,
                        help='encoder 后验位置模式：mlp=论文原方法，wls=几何增强')
    parser.add_argument('--normalize-obs', action='store_true',
                        help='仅归一化 encoder 输入的距离/锚点坐标，不引入额外监督')
    parser.add_argument('--no-detach-state', action='store_true',
                        help='Stage 1 内不截断 z_t 到下一步的梯度，使用完整 T 步反传')
    parser.add_argument('--use-range-mask', action='store_true',
                        help='mask 物理不可能的 UWB range；异常值用 anchor 中位数替换输入，loss 中忽略')
    parser.add_argument('--posterior-state-mode', choices=['sample', 'mean'], default=None,
                        help='后验状态用于重建/RNN的方式：sample=采样，mean=后验均值')
    parser.add_argument('--dt', type=float, default=None,
                        help='覆盖匀速运动模型时间步长')
    parser.add_argument('--decoder-max-std', type=float, default=None,
                        help='可选：限制 decoder 测距标准差上限，例如 0.3')
    parser.add_argument('--stage2-policy', choices=['actor', 'random', 'all'], default=None,
                        help='LocDreamer Stage 2 想象训练策略')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='覆盖检查点目录，建议每个实验单独目录')
    parser.add_argument('--tensorboard-dir', type=str, default=None,
                        help='TensorBoard log directory; default is checkpoint_dir/tensorboard')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard logging')
    parser.add_argument('--tb-eval-interval', type=int, default=None,
                        help='Run validation for TensorBoard every N epochs; default follows log_interval')
    parser.add_argument('--data-path', type=str, default=None,
                        help='覆盖预处理数据目录')
    parser.add_argument('--source-ids', type=str, default=None,
                        help='源 anchor 全局编号，例如 0,3,6')
    parser.add_argument('--target-ids', type=str, default=None,
                        help='目标 anchor 全局编号，例如 1,2,4,5,7；默认与 source 互补')
    args = parser.parse_args()

    # ── 配置 ──
    cfg = Config()
    if args.dssm_epochs is not None:
        cfg.dssm_epochs = args.dssm_epochs
    if args.imagine_epochs is not None:
        cfg.imagine_epochs = args.imagine_epochs
    if args.device is not None:
        cfg.device = args.device
    if args.pos_loss_weight is not None:
        cfg.pos_loss_weight = args.pos_loss_weight
    if args.scheduler_pos_reward_weight is not None:
        cfg.scheduler_pos_reward_weight = args.scheduler_pos_reward_weight
    if args.encoder_mode is not None:
        cfg.encoder_mode = args.encoder_mode
    if args.normalize_obs:
        cfg.normalize_obs = True
    if args.no_detach_state:
        cfg.detach_state_between_steps = False
    if args.use_range_mask:
        cfg.use_range_mask = True
    if args.posterior_state_mode is not None:
        cfg.posterior_state_mode = args.posterior_state_mode
    if args.dt is not None:
        cfg.dt = args.dt
    if args.decoder_max_std is not None:
        cfg.decoder_max_std = args.decoder_max_std
    if args.stage2_policy is not None:
        cfg.stage2_policy = args.stage2_policy
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.tb_eval_interval is not None:
        cfg.tensorboard_eval_interval = args.tb_eval_interval
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

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"[设备] 使用 {device}")
    print(f"[配置] 锚点总数 A={cfg.num_anchors}，每步调度 Kt={cfg.num_scheduled}，"
          f"源锚点 |R|={cfg.num_source}") 
    print(f"[配置] experiment={args.experiment}, encoder_mode={cfg.encoder_mode}, "
          f"normalize_obs={cfg.normalize_obs}, "
          f"posterior_state_mode={cfg.posterior_state_mode}, "
          f"stage2_policy={cfg.stage2_policy}, checkpoint_dir={cfg.checkpoint_dir}")
    print(f"[配置] source_anchor_ids={cfg.source_anchor_ids}, "
          f"target_anchor_ids={cfg.target_anchor_ids}, dt={cfg.dt}, "
          f"detach_state_between_steps={cfg.detach_state_between_steps}, "
          f"use_range_mask={cfg.use_range_mask}, decoder_max_std={cfg.decoder_max_std}")

    # ── 加载数据 ──
    data_loaders = get_data_loaders(cfg, use_synthetic=args.synthetic)
    source_anchor_pos = data_loaders['source_anchor_pos']
    target_anchor_pos = data_loaders['target_anchor_pos']
    print(f"[数据] 源锚点位置：{source_anchor_pos.shape}，"
          f"目标锚点位置：{target_anchor_pos.shape}")

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

    # 打印模型参数量
    dssm_params   = sum(p.numel() for p in dssm.parameters())
    actor_params  = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"[模型] DSSM: {dssm_params:,} 参数 | "
          f"Actor: {actor_params:,} 参数 | "
          f"Critic: {critic_params:,} 参数")

    writer = create_summary_writer(cfg, args)

    if args.experiment == 'dssm-all-real':
        # Table 2 baseline: DSSM - all anchors (real)。使用目标 5 个 anchors
        # 的真实训练测量，不进入想象训练和 Actor-Critic。
        dssm = pretrain_dssm(
            cfg, dssm, data_loaders, device,
            loader_key='target_train_loader',
            anchor_key='target_anchor_pos',
            checkpoint_name='dssm_all_real.pt',
            title='DSSM-all(real) baseline（目标5锚点真实数据）',
            writer=writer,
            tb_prefix='dssm_all_real',
            validate_source=False,
            validate_target_random=False,
        )
        torch.save(
            {'dssm': dssm.state_dict(), 'actor': actor.state_dict(), 'critic': critic.state_dict()},
            os.path.join(cfg.checkpoint_dir, 'locdreamer_best.pt')
        )
        print("\n[完成] DSSM-all(real) baseline 训练完毕，运行 evaluate.py --checkpoint best 评估。")
        if writer is not None:
            writer.flush()
            writer.close()
        return

    # ── Stage 1：DSSM 预训练 ──
    dssm = pretrain_dssm(
        cfg, dssm, data_loaders, device,
        writer=writer,
        tb_prefix='stage1',
        validate_source=True,
        validate_target_random=True,
    )

    if not args.dssm_only:
        # ── Stage 2：LocDreamer 想象训练 ──
        train_locdreamer(cfg, dssm, actor, critic, data_loaders, device, writer=writer)
        print("\n[完成] 模型训练完毕，运行 python evaluate.py 进行评估。")

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == '__main__':
    main()
