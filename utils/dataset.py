"""
数据集处理模块

对应论文 "Experiment Setup" 节：
  - 使用 Bregar 2023 室内 UWB 定位数据集（Scientific Data 10(1):744）
  - Environment 0：住宅楼层（9.18m × 12.06m），8个固定锚点，85个位置
  - 共 126410 个测量样本

数据集下载：
  https://zenodo.org/record/8060921
  下载后将数据文件放入 ./data/ 目录

实验设置：
  - 源环境（预训练）：使用 3 个锚点（R ⊂ {0,...,7}）的真实测量
  - 目标环境（想象训练和测试）：5 个新锚点 A = {a0,...,a4}（与 R 不同）
  - Kt=3 个锚点参与每步测距

若无真实数据，可使用本模块提供的合成数据生成器进行代码验证。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional


# =========================================================================
# 数据集配置（与实验设置对应）
# =========================================================================

# Environment 0 的空间范围（单位：米）
ENV0_WIDTH  = 9.18
ENV0_HEIGHT = 12.06

# 论文使用的锚点划分（示意，以 0-7 编号的 8 个锚点）
# 论文："use real measurements from three anchors to bootstrap the model,
#        then imagine for unseen set of five new anchors"
SOURCE_ANCHOR_IDS = [0, 1, 2]   # R：源环境 3 个锚点（预训练用）
TARGET_ANCHOR_IDS = [3, 4, 5, 6, 7]   # A：5 个新锚点（想象训练和评估用）


class UWBDataset(Dataset):
    """
    UWB 测量轨迹数据集

    将原始测量数据组织为定长序列（sequence_length=T），供模型训练。

    数据格式（每个样本）：
      - distances:    [T, num_anchors]  — 各时间步各锚点的测量距离
      - positions:    [T, 2]           — 各时间步目标真实位置 [x, y]

    Args:
        distances:       [N, num_anchors]  — 所有时间步的距离测量
        positions:       [N, 2]           — 所有时间步的目标位置
        anchor_positions: [num_anchors, 2] — 锚点位置（固定）
        seq_len:         序列长度 T（滑动窗口切片）
        stride:          滑动步长（默认 = seq_len，不重叠）
    """

    def __init__(self, distances: np.ndarray, positions: np.ndarray,
                 anchor_positions: np.ndarray, seq_len: int = 32,
                 stride: int = None, trajectory_ids: np.ndarray = None,
                 distance_masks: np.ndarray = None):
        self.distances = torch.tensor(distances, dtype=torch.float32)
        self.positions = torch.tensor(positions, dtype=torch.float32)
        self.anchor_positions = torch.tensor(anchor_positions, dtype=torch.float32)
        self.distance_masks = None
        if distance_masks is not None:
            self.distance_masks = torch.tensor(distance_masks, dtype=torch.float32)
        self.trajectory_ids = None
        if trajectory_ids is not None:
            self.trajectory_ids = torch.tensor(trajectory_ids, dtype=torch.long)

        self.seq_len = seq_len
        self.stride = stride or seq_len   # 默认不重叠切片

        # 计算可用的序列起始索引
        N = len(distances)
        if self.trajectory_ids is None:
            self.indices = list(range(0, N - seq_len + 1, self.stride))
        else:
            ids = np.asarray(trajectory_ids)
            self.indices = []
            start = 0
            while start < N:
                end = start + 1
                while end < N and ids[end] == ids[start]:
                    end += 1
                self.indices.extend(range(start, end - seq_len + 1, self.stride))
                start = end

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.seq_len

        sample = {
            # [T, num_anchors]：该序列内各时间步、各锚点的距离测量
            'distances': self.distances[start:end],
            # [T, 2]：对应的真实位置（仅用于评估，不用于训练损失）
            'positions': self.positions[start:end],
        }
        if self.distance_masks is not None:
            sample['distance_masks'] = self.distance_masks[start:end]
        return sample


def load_uwb_data(data_dir: str,
                  source_ids: List[int] = None,
                  target_ids: List[int] = None,
                  seq_len: int = 32,
                  train_ratio: float = 0.8,
                  seed: int = 42,
                  use_range_mask: bool = False,
                  min_valid_range: float = 0.0,
                  max_valid_range: float = 30.0
                  ) -> Dict[str, object]:
    """
    加载 Bregar 2023 UWB 数据集（Environment 0）。

    预期数据目录结构（数据集下载后预处理为 .npy 格式）：
      data_dir/
        anchor_positions.npy   — [8, 2]，8 个锚点的 (x, y) 坐标（米）
        positions.npy          — [N, 2]，目标真实位置序列
        distances.npy          — [N, 8]，8 个锚点的距离测量序列

    如果原始数据集格式不同，请参考 preprocess_bregar_dataset() 函数预处理。

    Args:
        data_dir:     数据目录路径
        source_ids:   源锚点编号列表（用于预训练），默认 [0,1,2]
        target_ids:   目标锚点编号列表（用于想象训练和评估），默认 [3,4,5,6,7]
        seq_len:      序列长度 T=32（Table 1）
        train_ratio:  训练集比例
        seed:         随机种子

    Returns:
        字典包含：
          'source_train_loader':  预训练用数据加载器（源锚点测量）
          'source_val_loader':    源环境验证集
          'target_val_loader':    目标环境测试集（用于评估）
          'anchor_positions_all': [8, 2]，所有锚点位置
          'source_positions':     [|R|, 2]，源锚点位置
          'target_positions':     [|A|, 2]，目标锚点位置
    """
    source_ids = source_ids or SOURCE_ANCHOR_IDS
    target_ids = target_ids or TARGET_ANCHOR_IDS

    # ── 加载预处理后的 .npy 文件 ──
    anchor_pos_path = os.path.join(data_dir, 'anchor_positions.npy')
    positions_path  = os.path.join(data_dir, 'positions.npy')
    distances_path  = os.path.join(data_dir, 'distances.npy')

    if not all(os.path.exists(p) for p in [anchor_pos_path, positions_path, distances_path]):
        raise FileNotFoundError(
            f"数据文件不存在于 {data_dir}。\n"
            "请先下载 Bregar 2023 UWB 数据集并运行 preprocess_bregar_dataset()。\n"
            "下载地址：https://zenodo.org/record/8060921\n"
            "或使用 generate_synthetic_data() 生成合成数据进行测试。"
        )

    anchor_positions = np.load(anchor_pos_path)   # [8, 2]
    positions        = np.load(positions_path)    # [N, 2]
    distances        = np.load(distances_path)    # [N, 8]
    trajectory_ids_path = os.path.join(data_dir, 'trajectory_ids.npy')
    trajectory_ids = np.load(trajectory_ids_path) if os.path.exists(trajectory_ids_path) else None

    # ── 划分训练/验证集 ──
    N = len(positions)
    np.random.seed(seed)
    split = int(N * train_ratio)
    # 按时间顺序切分（不打乱，保持轨迹连续性）
    train_dist = distances[:split]
    train_pos  = positions[:split]
    val_dist   = distances[split:]
    val_pos    = positions[split:]
    train_traj = trajectory_ids[:split] if trajectory_ids is not None else None
    val_traj   = trajectory_ids[split:] if trajectory_ids is not None else None

    train_mask = val_mask = None
    if use_range_mask:
        valid = (
            np.isfinite(distances)
            & (distances > min_valid_range)
            & (distances < max_valid_range)
        )
        train_valid = valid[:split]
        train_for_median = np.where(train_valid, train_dist, np.nan)
        medians = np.nanmedian(train_for_median, axis=0)
        global_median = np.nanmedian(train_for_median)
        medians = np.where(np.isfinite(medians), medians, global_median)
        distances = np.where(valid, distances, medians[None, :]).astype(np.float32)
        train_dist = distances[:split]
        val_dist = distances[split:]
        train_mask = valid[:split].astype(np.float32)
        val_mask = valid[split:].astype(np.float32)

    # ── 提取源/目标锚点测量和位置 ──
    source_anchor_pos = anchor_positions[source_ids]    # [|R|, 2]
    target_anchor_pos = anchor_positions[target_ids]    # [|A|, 2]

    # 源环境：只使用源锚点的距离测量（预训练）
    source_train_dist = train_dist[:, source_ids]       # [N_train, |R|]
    source_val_dist   = val_dist[:, source_ids]         # [N_val, |R|]
    source_train_mask = train_mask[:, source_ids] if train_mask is not None else None
    source_val_mask = val_mask[:, source_ids] if val_mask is not None else None

    # 目标环境：使用目标锚点的距离测量（仅用于评估）
    target_train_dist = train_dist[:, target_ids]       # [N_train, |A|]
    target_val_dist   = val_dist[:, target_ids]         # [N_val, |A|]
    target_train_mask = train_mask[:, target_ids] if train_mask is not None else None
    target_val_mask = val_mask[:, target_ids] if val_mask is not None else None

    # ── 构建 PyTorch Dataset ──
    source_train_dataset = UWBDataset(
        source_train_dist, train_pos, source_anchor_pos, seq_len,
        trajectory_ids=train_traj,
        distance_masks=source_train_mask,
    )
    source_val_dataset = UWBDataset(
        source_val_dist, val_pos, source_anchor_pos, seq_len,
        trajectory_ids=val_traj,
        distance_masks=source_val_mask,
    )
    target_train_dataset = UWBDataset(
        target_train_dist, train_pos, target_anchor_pos, seq_len,
        trajectory_ids=train_traj,
        distance_masks=target_train_mask,
    )
    target_val_dataset = UWBDataset(
        target_val_dist, val_pos, target_anchor_pos, seq_len,
        trajectory_ids=val_traj,
        distance_masks=target_val_mask,
    )

    return {
        'source_train_loader': DataLoader(
            source_train_dataset, batch_size=32, shuffle=True, drop_last=True
        ),
        'source_val_loader': DataLoader(
            source_val_dataset, batch_size=32, shuffle=False
        ),
        'target_train_loader': DataLoader(
            target_train_dataset, batch_size=32, shuffle=True, drop_last=True
        ),
        'target_val_loader': DataLoader(
            target_val_dataset, batch_size=32, shuffle=False
        ),
        'anchor_positions_all': torch.tensor(anchor_positions, dtype=torch.float32),
        'source_anchor_pos':    torch.tensor(source_anchor_pos, dtype=torch.float32),
        'target_anchor_pos':    torch.tensor(target_anchor_pos, dtype=torch.float32),
    }


def generate_synthetic_data(num_trajectories: int = 200,
                             traj_len: int = 200,
                             env_size: Tuple[float, float] = (9.18, 12.06),
                             num_anchors: int = 8,
                             noise_std: float = 0.3,
                             nlos_prob: float = 0.2,
                             seed: int = 42
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成合成 UWB 数据（用于无真实数据时的代码测试）

    运动模型：匀速运动 + 随机加速度扰动（类似真实行人轨迹）
    测量模型：真实距离 + 高斯噪声 + NLoS 偏置

    Args:
        num_trajectories: 轨迹数量
        traj_len:         每条轨迹的时间步数
        env_size:         环境尺寸 (width, height)（米）
        num_anchors:      锚点数量
        noise_std:        测量噪声标准差（米）
        nlos_prob:        NLoS 偏置概率（模拟非视距传播）
        seed:             随机种子

    Returns:
        positions:        [N, 2]          — 所有时间步的目标位置
        distances:        [N, num_anchors] — 各锚点距离测量
        anchor_positions: [num_anchors, 2] — 锚点位置
    """
    np.random.seed(seed)
    W, H = env_size

    # ── 固定锚点位置（均匀分布在环境边界上）──
    # 8 个锚点，类似论文 Environment 0 的部署方式
    if num_anchors == 8:
        anchor_positions = np.array([
            [0.0, 0.0],          # 左下角
            [W/2, 0.0],          # 底部中间
            [W,   0.0],          # 右下角
            [W,   H/3],          # 右侧 1/3
            [W,   2*H/3],        # 右侧 2/3
            [W,   H],            # 右上角
            [W/2, H],            # 顶部中间
            [0.0, H],            # 左上角
        ], dtype=np.float32)
    else:
        # 随机部署
        anchor_positions = np.column_stack([
            np.random.uniform(0, W, num_anchors),
            np.random.uniform(0, H, num_anchors),
        ]).astype(np.float32)

    all_positions = []
    all_distances = []

    for _ in range(num_trajectories):
        # ── 生成随机游走轨迹（匀速 + 随机扰动）──
        pos = np.array([
            np.random.uniform(0.5, W - 0.5),
            np.random.uniform(0.5, H - 0.5),
        ], dtype=np.float32)

        # 初始速度（慢速行走，约 0.5-1.5 m/s）
        vel = np.random.randn(2).astype(np.float32) * 0.5
        dt = 0.1  # 时间步长 0.1 秒

        traj_pos = []
        traj_dist = []

        for t in range(traj_len):
            # 随机加速度（模拟转向、停顿）
            acc = np.random.randn(2).astype(np.float32) * 0.3
            vel = vel + acc * dt
            # 限制速度（最大 2 m/s）
            speed = np.linalg.norm(vel)
            if speed > 2.0:
                vel = vel / speed * 2.0

            pos = pos + vel * dt
            # 边界反弹（防止走出环境）
            for dim, limit in enumerate([W, H]):
                if pos[dim] < 0.1:
                    pos[dim] = 0.1
                    vel[dim] = abs(vel[dim])
                elif pos[dim] > limit - 0.1:
                    pos[dim] = limit - 0.1
                    vel[dim] = -abs(vel[dim])

            traj_pos.append(pos.copy())

            # ── 生成距离测量（式 1：d_t^k = ||p_t - p^k|| + n_t^k）──
            dist_row = np.zeros(num_anchors, dtype=np.float32) #（8，）
            for k in range(num_anchors):
                # 真实距离
                true_dist = np.linalg.norm(pos - anchor_positions[k])
                # 测量噪声（高斯）
                noise = np.random.randn() * noise_std
                # NLoS 偏置（部分测量有额外正偏置，模拟反射路径）
                nlos_bias = np.random.exponential(0.5) if np.random.rand() < nlos_prob else 0.0
                dist_row[k] = max(true_dist + noise + nlos_bias, 0.01)  # 距离非负

            traj_dist.append(dist_row)

        all_positions.extend(traj_pos)
        all_distances.extend(traj_dist)

    positions = np.array(all_positions, dtype=np.float32)    # [N, 2] N=40000
    distances = np.array(all_distances, dtype=np.float32)    # [N, num_anchors]

    return positions, distances, anchor_positions #anchor_positions: (8,2)


def preprocess_bregar_dataset(raw_data_path: str, output_dir: str,
                              min_range: float = None,
                              max_range: float = None):
    """
    预处理 Bregar 2023 原始数据集，转换为模型所需的 .npy 格式。

    Bregar 2023 数据集格式（Scientific Data 2023）：
      - 数据集存储为 CSV 或 MAT 文件
      - 包含：锚点位置、目标位置、各锚点的 UWB 测量值（CIR 或 range）

    使用前请先从 https://zenodo.org/record/8060921 下载数据集，
    然后根据实际文件格式修改此函数。

    Args:
        raw_data_path: 原始数据集路径
        output_dir:    输出目录（存储预处理后的 .npy 文件）
    """
    import csv

    def resolve_paths(path: str) -> Tuple[str, str]:
        path = os.path.abspath(path)
        candidates = [
            path,
            os.path.join(path, 'environment0'),
            os.path.join(path, 'data_set', 'environment0'),
        ]
        for env_dir in candidates:
            if os.path.exists(os.path.join(env_dir, 'anchors.csv')) and \
               os.path.exists(os.path.join(env_dir, 'walking_path.csv')):
                root = os.path.dirname(env_dir)
                raw_env = os.path.join(root, 'raw_data', 'environment0')
                if os.path.exists(os.path.join(raw_env, 'data')):
                    return env_dir, raw_env
        raise FileNotFoundError(
            f"无法在 {raw_data_path} 下定位 Bregar Environment 0。"
        )

    def read_ranges(path: str) -> List[float]:
        ranges = []
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            next(f)  # version
            next(f)  # header
            for line in f:
                parts = line.rstrip('\n').split(',', 10)
                if len(parts) > 9:
                    ranges.append(float(parts[9]))
        return ranges

    env_dir, raw_env_dir = resolve_paths(raw_data_path)
    os.makedirs(output_dir, exist_ok=True)

    anchor_rows = []
    with open(os.path.join(env_dir, 'anchors.csv'), newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            anchor_rows.append((row['Anchor'], float(row['x']), float(row['y'])))
    anchor_rows.sort(key=lambda x: int(x[0][1:]))
    anchor_names = [row[0] for row in anchor_rows]
    anchor_positions = np.array([[row[1], row[2]] for row in anchor_rows], dtype=np.float32)

    path_rows = []
    with open(os.path.join(env_dir, 'walking_path.csv'), newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x, y, z = float(row['x']), float(row['y']), float(row['z'])
            path_rows.append((x, y, z, f"{x:.2f}_{y:.2f}_{z:.2f}"))

    channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch7']
    data_root = os.path.join(raw_env_dir, 'data')
    all_positions = []
    all_distances = []
    trajectory_ids = []
    trajectory_id = 0

    dropped_rows = 0
    dropped_values = 0

    for ch in channels:
        per_location = []
        for _, _, _, loc_name in path_rows:
            anchor_ranges = []
            for anchor in anchor_names:
                csv_path = os.path.join(data_root, loc_name, f'{ch}_{anchor}.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(csv_path)
                anchor_ranges.append(read_ranges(csv_path))
            lengths = {len(r) for r in anchor_ranges}
            if len(lengths) != 1:
                raise ValueError(f"{loc_name}/{ch} 各锚点样本数不一致: {sorted(lengths)}")
            per_location.append(anchor_ranges)

        num_repeats = len(per_location[0][0])
        for repeat_idx in range(num_repeats):
            for pos_idx, (x, y, _, _) in enumerate(path_rows):
                row_distances = [
                    per_location[pos_idx][anchor_idx][repeat_idx]
                    for anchor_idx in range(len(anchor_names))
                ]
                invalid_count = 0
                if min_range is not None or max_range is not None:
                    invalid_count = sum(
                        (not np.isfinite(value))
                        or (min_range is not None and value <= min_range)
                        or (max_range is not None and value > max_range)
                        for value in row_distances
                    )
                if invalid_count:
                    dropped_rows += 1
                    dropped_values += invalid_count
                    continue
                all_positions.append([x, y])
                all_distances.append(row_distances)
                trajectory_ids.append(trajectory_id)
            trajectory_id += 1

    positions = np.asarray(all_positions, dtype=np.float32)
    distances = np.asarray(all_distances, dtype=np.float32)
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)

    np.save(os.path.join(output_dir, 'anchor_positions.npy'), anchor_positions)
    np.save(os.path.join(output_dir, 'positions.npy'), positions)
    np.save(os.path.join(output_dir, 'distances.npy'), distances)
    np.save(os.path.join(output_dir, 'trajectory_ids.npy'), trajectory_ids)

    print(f"[预处理] Environment 0: anchors={anchor_positions.shape}, "
          f"positions={positions.shape}, distances={distances.shape}, "
          f"trajectories={trajectory_id}, dropped_rows={dropped_rows}, "
          f"dropped_invalid_values={dropped_values}")


def get_data_loaders(cfg, use_synthetic: bool = False
                     ) -> Dict[str, object]:
    """
    统一接口：获取数据加载器（真实数据或合成数据）

    Args:
        cfg:           Config 对象
        use_synthetic: True → 使用合成数据（适合快速测试）

    Returns:
        与 load_uwb_data() 相同格式的字典
    """
    if use_synthetic:
        print("[数据] 使用合成数据（适合代码验证）...")
        positions, distances, anchor_positions = generate_synthetic_data(
            num_trajectories=200,
            traj_len=200,
            seed=cfg.seed,
        )

        # 与真实数据相同的划分逻辑
        N = len(positions)
        split = int(N * 0.8)

        source_ids = list(range(cfg.num_source))         # 前 3 个锚点作为源
        target_ids = list(range(cfg.num_source, cfg.num_source + cfg.num_anchors))  # 后 5 个
        if hasattr(cfg, 'source_anchor_ids') and cfg.source_anchor_ids is not None:
            source_ids = list(cfg.source_anchor_ids)
        if hasattr(cfg, 'target_anchor_ids') and cfg.target_anchor_ids is not None:
            target_ids = list(cfg.target_anchor_ids)

        source_anchor_pos = anchor_positions[source_ids]
        target_anchor_pos = anchor_positions[target_ids]

        source_train_ds = UWBDataset(
            distances[:split, source_ids], positions[:split],
            source_anchor_pos, cfg.seq_len
        )
        source_val_ds = UWBDataset(
            distances[split:, source_ids], positions[split:],
            source_anchor_pos, cfg.seq_len
        )
        target_val_ds = UWBDataset(
            distances[split:, target_ids], positions[split:],
            target_anchor_pos, cfg.seq_len
        )
        target_train_ds = UWBDataset(
            distances[:split, target_ids], positions[:split],
            target_anchor_pos, cfg.seq_len
        )

        return {
            'source_train_loader': DataLoader(
                source_train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True
            ),
            'source_val_loader': DataLoader(
                source_val_ds, batch_size=cfg.batch_size, shuffle=False
            ),
            'target_train_loader': DataLoader(
                target_train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True
            ),
            'target_val_loader': DataLoader(
                target_val_ds, batch_size=cfg.batch_size, shuffle=False
            ),
            'anchor_positions_all': torch.tensor(anchor_positions, dtype=torch.float32),
            'source_anchor_pos':    torch.tensor(source_anchor_pos, dtype=torch.float32),
            'target_anchor_pos':    torch.tensor(target_anchor_pos, dtype=torch.float32),
        }
    else:
        return load_uwb_data(
            cfg.data_path,
            source_ids=list(cfg.source_anchor_ids),
            target_ids=list(cfg.target_anchor_ids),
            seq_len=cfg.seq_len,
            train_ratio=0.8,
            seed=cfg.seed,
            use_range_mask=getattr(cfg, 'use_range_mask', False),
            min_valid_range=getattr(cfg, 'min_valid_range', 0.0),
            max_valid_range=getattr(cfg, 'max_valid_range', 30.0),
        )
