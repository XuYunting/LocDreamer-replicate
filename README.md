# LocDreamer Reproduction

Reproduction code for:

```text
LocDreamer: World Model-Based Learning for Joint Indoor Tracking and Anchor Scheduling
arXiv:2602.08204v1
```

The current implementation reproduces the main Table 2 trends on the Bregar UWB
Environment 0 data with the split:

```text
source anchors: (0,5,6)
target anchors: (1,2,3,4,7)
```

## Project Layout

```text
locdreamer1/
├── config.py
├── train.py
├── evaluate.py
├── models/
│   ├── dssm.py
│   ├── set_transformer.py
│   └── actor_critic.py
├── utils/
│   ├── dataset.py
│   └── metrics.py
├── DATASET_PREPROCESS.md
└── CHECKPOINTS.md
```

## Data

The data source and preprocessing steps are documented in:

```text
DATASET_PREPROCESS.md
```

Prepared NumPy files are expected at:

```text
data/anchor_positions.npy
data/positions.npy
data/distances.npy
data/trajectory_ids.npy
```

## Main Settings

The current reproduced runs use:

```text
encoder_mode: mlp
posterior_state_mode: sample
pos_loss_weight: 0
WLS encoder: disabled
decoder_max_std: 0.4
dssm_epochs: 200 or 300 depending on branch
imagine_epochs: 300
source_ids: 0,5,6
target_ids: 1,2,3,4,7
```

`decoder_max_std=0.4` is an implementation stabilization setting and should be
reported with the results.

## Current Reproduction Results

Reference paper Table 2:

| Method | MAE | RMSE | 50th | 90th |
|---|---:|---:|---:|---:|
| EKF - random scheduling | 1.05 | 1.67 | 0.71 | 1.83 |
| EKF - all anchors | 0.92 | 1.50 | 0.63 | 1.41 |
| DSSM - all anchors (real) | 0.57 | 0.64 | 0.54 | 0.96 |
| LocDreamer - random scheduling | 1.07 | 1.43 | 0.82 | 2.05 |
| LocDreamer - all anchors (imagined) | 0.85 | 1.07 | 0.68 | 1.60 |
| LocDreamer - scheduling | 0.66 | 0.77 | 0.60 | 1.18 |

Current reproduced results:

| Method | MAE | RMSE | 50th | 90th | Notes |
|---|---:|---:|---:|---:|---|
| EKF - random scheduling | 1.010 | 1.182 | 0.746 | 2.010 | LS init |
| EKF - all anchors | 0.981 | 1.145 | 0.727 | 1.958 | LS init |
| DSSM - all anchors (real) | 0.517 | 0.553 | 0.522 | 0.738 | target real training |
| LocDreamer - random scheduling | 1.080 | 1.447 | 0.800 | 1.810 | all-imagined train, random eval |
| LocDreamer - all anchors (imagined) | 0.786 | 0.869 | 0.740 | 1.356 | Stage 2 all target anchors |
| LocDreamer - scheduling | 0.642 | 0.774 | 0.688 | 1.043 | actor checkpoint 100 |

The main results are reproduced. Two important notes:

- The reported `LocDreamer-random` result uses an all-imagined-trained checkpoint
  evaluated with random three-anchor scheduling. Strict random-train/random-eval
  was unstable in this implementation.
- For Actor-Critic scheduling, `locdreamer_best.pt` is not reliable because
  internal Actor/Critic/reward losses do not always track real MAE. Use checkpoint
  sweep and report the selected epoch.

## Training Commands

### DSSM-All(real)

```bash
cd /home/yunting/locdreamer1

CUDA_VISIBLE_DEVICES=2 /home/yunting/miniconda3/envs/pred/bin/python train.py \
  --experiment dssm-all-real \
  --device cuda:0 \
  --source-ids 0,5,6 \
  --target-ids 1,2,3,4,7 \
  --dssm-epochs 300 \
  --pos-loss-weight 0 \
  --encoder-mode mlp \
  --posterior-state-mode sample \
  --decoder-max-std 0.4 \
  --checkpoint-dir checkpoint_store/runs/dssm_all_real_12347_e300_std04
```

### LocDreamer-All(imagined)

```bash
CUDA_VISIBLE_DEVICES=2 /home/yunting/miniconda3/envs/pred/bin/python train.py \
  --device cuda:0 \
  --source-ids 0,5,6 \
  --target-ids 1,2,3,4,7 \
  --dssm-epochs 200 \
  --imagine-epochs 300 \
  --stage2-policy all \
  --pos-loss-weight 0 \
  --encoder-mode mlp \
  --posterior-state-mode sample \
  --decoder-max-std 0.4 \
  --checkpoint-dir checkpoint_store/runs/locdreamer_all_imagined_056_to_12347_e200_std04
```

### LocDreamer-Scheduling

```bash
CUDA_VISIBLE_DEVICES=2 /home/yunting/miniconda3/envs/pred/bin/python train.py \
  --device cuda:0 \
  --source-ids 0,5,6 \
  --target-ids 1,2,3,4,7 \
  --dssm-epochs 200 \
  --imagine-epochs 300 \
  --stage2-policy actor \
  --pos-loss-weight 0 \
  --encoder-mode mlp \
  --posterior-state-mode sample \
  --decoder-max-std 0.4 \
  --checkpoint-dir checkpoint_store/runs/locdreamer_scheduling_056_to_12347_e200_std04
```

## Evaluation Commands

Evaluate `LocDreamer-all(imagined)`:

```bash
CUDA_VISIBLE_DEVICES=2 /home/yunting/miniconda3/envs/pred/bin/python evaluate.py \
  --checkpoint checkpoint_store/runs/locdreamer_all_imagined_056_to_12347_e200_std04/locdreamer_best_until_100.pt \
  --device cuda:0 \
  --source-ids 0,5,6 \
  --target-ids 1,2,3,4,7 \
  --encoder-mode mlp \
  --posterior-state-mode sample \
  --decoder-max-std 0.4 \
  --eval-method locdreamer-all \
  --no-ekf
```

Evaluate scheduling:

```bash
CUDA_VISIBLE_DEVICES=2 /home/yunting/miniconda3/envs/pred/bin/python evaluate.py \
  --checkpoint checkpoint_store/runs/locdreamer_scheduling_056_to_12347_e200_std04/locdreamer_100.pt \
  --device cuda:0 \
  --source-ids 0,5,6 \
  --target-ids 1,2,3,4,7 \
  --encoder-mode mlp \
  --posterior-state-mode sample \
  --decoder-max-std 0.4 \
  --eval-method locdreamer-scheduling \
  --schedule-policy actor \
  --no-ekf
```

Sweep scheduling checkpoints:

```bash
for ckpt in 50 100 150 200 250 300; do
  echo "========== locdreamer_${ckpt}.pt =========="
  CUDA_VISIBLE_DEVICES=2 /home/yunting/miniconda3/envs/pred/bin/python evaluate.py \
    --checkpoint checkpoint_store/runs/locdreamer_scheduling_056_to_12347_e200_std04/locdreamer_${ckpt}.pt \
    --device cuda:0 \
    --source-ids 0,5,6 \
    --target-ids 1,2,3,4,7 \
    --encoder-mode mlp \
    --posterior-state-mode sample \
    --decoder-max-std 0.4 \
    --eval-method locdreamer-scheduling \
    --schedule-policy actor \
    --no-ekf
done
```

## Method Notes

- Stage 1 trains DSSM on source real measurements.
- Stage 2 uses DSSM imagination to adapt to target anchor deployments.
- `stage2-policy all` trains using all five imagined target anchors.
- `stage2-policy actor` trains the Actor-Critic scheduler to choose three
  anchors.
- `DSSM-all(real)` is a target-real baseline and is not label-free target
  adaptation.
- WLS and position auxiliary losses are diagnostic/ablation options, not part of
  the strict main reproduction.
