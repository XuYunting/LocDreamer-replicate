# LocDreamer Dataset Download and Preprocessing Notes

This note records how the Bregar UWB dataset was downloaded and converted for
the current LocDreamer reproduction. Training and evaluation results are kept in
`README.md` and `CHECKPOINTS.md`.

## Source

Dataset:

```text
Bregar, K. (2023). Indoor UWB positioning and position tracking data set.
Zenodo DOI: 10.5281/zenodo.8280736
Zenodo page: https://zenodo.org/records/8280736
```

Download URL:

```bash
https://zenodo.org/records/8280736/files/data_set.zip?download=1
```

Expected/observed MD5:

```text
030cce5aadeab7b1b3dbd6901a49bfc2
```

## Local Paths

Downloaded zip:

```text
/home/yunting/dataset/data_set.zip
```

Extracted dataset:

```text
/home/yunting/dataset/data_set
```

LocDreamer-ready NumPy files:

```text
/home/yunting/locdreamer1/data/anchor_positions.npy
/home/yunting/locdreamer1/data/positions.npy
/home/yunting/locdreamer1/data/distances.npy
/home/yunting/locdreamer1/data/trajectory_ids.npy
```

## Download and Extract

```bash
mkdir -p /home/yunting/dataset
cd /home/yunting/dataset

curl -L -C - -o data_set.zip \
  "https://zenodo.org/records/8280736/files/data_set.zip?download=1"

md5sum data_set.zip
unzip -q data_set.zip -d /home/yunting/dataset
```

The zip contains a top-level `data_set/` directory, so extracting into
`/home/yunting/dataset` is safe.

## Dataset Structure Used

The reproduction uses Environment 0:

```text
/home/yunting/dataset/data_set/environment0/anchors.csv
/home/yunting/dataset/data_set/environment0/walking_path.csv
/home/yunting/dataset/data_set/raw_data/environment0/data/*/ch*_A*.csv
```

Environment 0 contains:

```text
8 anchors: A1 to A8
6 UWB channels: ch1, ch2, ch3, ch4, ch5, ch7
85 walking-path positions
```

Only the `RANGE` column is used from each raw CSV.

## Preprocessing Logic

Implementation:

```text
/home/yunting/locdreamer1/utils/dataset.py
```

Function:

```python
preprocess_bregar_dataset(raw_data_path, output_dir)
```

The preprocessing:

1. Reads 2D anchor positions from `environment0/anchors.csv`.
2. Reads walking-path positions from `environment0/walking_path.csv`.
3. Iterates through channels and walking-path folders.
4. Reads `chX_A1.csv` through `chX_A8.csv`.
5. Extracts aligned `RANGE` rows across all anchors.
6. Saves NumPy arrays for anchors, positions, distances, and trajectory ids.

Generated shapes:

```text
anchor_positions.npy: (8, 2)
positions.npy:        (15810, 2)
distances.npy:        (15810, 8)
trajectory_ids.npy:   (15810,)
trajectories:         186
```

This corresponds to:

```text
15810 rows * 8 anchors = 126480 individual range measurements
```

The paper reports 126410 measurements for Environment 0. The small difference is
likely due to a filtering/counting convention; this preprocessing uses all
aligned raw range rows available in the extracted Environment 0 files.

## Re-run Preprocessing

```bash
cd /home/yunting/locdreamer1

/home/yunting/miniconda3/bin/conda run -n pred python -c \
  "from utils.dataset import preprocess_bregar_dataset; preprocess_bregar_dataset('/home/yunting/dataset/data_set', '/home/yunting/locdreamer1/data')"
```

## Data Loader Check

The current main split is:

```text
source anchors: (0,5,6)
target anchors: (1,2,3,4,7)
scheduled anchors Kt: 3
sequence length T: 32
```

Validation command:

```bash
cd /home/yunting/locdreamer1

/home/yunting/miniconda3/bin/conda run -n pred python -c \
  "from config import Config; from utils.dataset import get_data_loaders; cfg=Config(); d=get_data_loaders(cfg); print(d['source_anchor_pos'].shape, d['target_anchor_pos'].shape); print(len(d['source_train_loader'].dataset), len(d['source_val_loader'].dataset), len(d['target_val_loader'].dataset)); b=next(iter(d['source_train_loader'])); print(b['distances'].shape, b['positions'].shape)"
```

Observed output:

```text
torch.Size([3, 2]) torch.Size([5, 2])
298 74 74
torch.Size([32, 32, 3]) torch.Size([32, 32, 2])
```

## Reproduction Notes

Current training and evaluation results are documented in:

```text
README.md
CHECKPOINTS.md
```

Older exploratory runs, WLS/position-aided diagnostics, failed intermediate
experiments, and anchor-split scans were removed from this data note to avoid
confusing them with the current reproduction path.

## Disk Space Note

The extracted dataset is large:

```text
/home/yunting/dataset/data_set      about 14G
/home/yunting/dataset/data_set.zip  about 4.2G
```

After successful extraction and MD5 verification, the zip can be deleted to
recover space:

```bash
rm /home/yunting/dataset/data_set.zip
```
