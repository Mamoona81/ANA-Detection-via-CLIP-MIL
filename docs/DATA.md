# Data Documentation

## Dataset Overview

**ANA Fluorescence Microscopy with CLIP Embeddings**
- **Total Samples:** ~6,500 ANA images
- **Indexed Samples:** 4,000+ with labels
- **Feature Type:** CLIP ViT-L/14 embeddings
- **Feature Dimension:** [257, 768] per sample (1 CLS + 256 patches)
- **Classes:** 8 ICAP-standardized ANA patterns (multi-label)
- **Data Split:** Train/Val/Test via CSV index

## File Organization

### Directory Structure

```
data/
├── features_index_english.csv          # CSV index (~4000 rows)
├── *.npy                                # Individual CLIP embeddings
├── samples/                             # Sample subset (50-100 files for demo)
│   └── *.npy
└── README.md                            # This file
```

### File Sizes

| Item | Size |
|------|------|
| Single .npy file | 0.8 MB - 5 MB (variable) |
| features_index_english.csv | ~2 MB |
| Full dataset (~6500 files) | ~100+ GB |
| Sample subset (~75 files) | ~300 MB |

## CSV Index Format

### Columns

| Index | Name | Type | Example | Notes |
|-------|------|------|---------|-------|
| 0 | patient_id | str | 022541326800 | Unique patient identifier |
| 1 | feature_file | str | 022541326800_20180810B1-2_1：320_peripheralcytoplasm.npy | Path to .npy file |
| 2 | date | str | 20180810 | Date of microscopy |
| 3 | titer | str | 1:320 | Antibody titer (dilution) |
| 4 | specimen_type | str | serum | Sample type |
| 5-7 | reserved | - | - | Not used |
| 8 | Split | str | train/val/test | Dataset partition |
| 9-14 | reserved | - | - | Metadata |
| **15-22** | **ICAP Classes** | **binary** | **0 or 1** | **Multi-label targets** |

### ICAP Class Labels (Columns 15-22, 0-indexed)

| Index | Class Name | Full Name |
|-------|-----------|-----------|
| 15 | golgi_apparatus | Golgi Apparatus |
| 16 | homogeneous | Homogeneous |
| 17 | nucleolus | Nucleolus |
| 18 | nuclear_dot | Nuclear Dot |
| 19 | centromere | Centromere |
| 20 | nuclear_membrane_peripheral | Nuclear Membrane (Peripheral) |
| 21 | cytoplasm_mitochondria | Cytoplasm/Mitochondria |
| 22 | speckled_granular_dfs70 | Speckled Granular (DFS70) |

### Sample CSV Row

```
022541326800,022541326800_20180810B1-2_1：320_peripheralcytoplasm.npy,20180810,1:320,serum,...,train,0,0,0,1,0,1,0,1,1
↓                                                                                       ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
patient_id                           feature_file                                  split [golgi, homog, nucleolus, nuclear_dot, centromere, nuc_mem, cytoplasm, speckled]
```

In this sample:
- **Split:** train
- **Labels:** nuclear_dot=1, centromere=1, cytoplasm_mitochondria=1, speckled_granular_dfs70=1
- **Multi-label:** 4 ANA patterns present

## CLIP Embedding Structure

### File Format

- **Type:** NumPy array (.npy binary format)
- **Shape:** (257, 768)
- **Dtype:** float32
- **Format:** NumPy's native binary format (portable across platforms)

### Loading & Inspection

```python
import numpy as np

# Load embedding
embedding = np.load("data/022541326800_20180810B1-2_1：320_peripheralcytoplasm.npy")
print(embedding.shape)  # (257, 768)
print(embedding.dtype)  # float32

# Extract CLS token (global representation)
cls_token = embedding[0]  # Shape: (768,)

# Patches (local representations)
patches = embedding[1:]  # Shape: (256, 768)

# Normalization check
norm = np.linalg.norm(embedding[0])  # Should be ~1.0 if L2-normalized
print(f"CLS norm: {norm:.4f}")
```

### Semantic Interpretation

- **CLS Token (Row 0):** CLIP's global image representation
  - Captures overall image content
  - Used as primary feature in MIL model
  
- **Patch Tokens (Rows 1-256):** Local image regions
  - Each row is a 14×14 pixel region embedding
  - 16×16 grid of patches over image
  - Not directly used in baseline MIL but available for future work

## Data Split Strategy

### Split Distribution

```
Train:  2,850 samples (71%)
Val:    575 samples (14%)
Test:   575 samples (15%)
────────────────────────
Total:  4,000 samples
```

### Stratification

- **By class:** Multi-label stratification attempted (no label leakage)
- **Temporal:** Samples from different dates distributed across splits
- **Patient-level:** Attempt to avoid data leakage (same patient in single split)

### Checking Splits

```python
import pandas as pd

df = pd.read_csv("data/features_index_english.csv")

# Count by split
print(df['Split'].value_counts())
# train    2850
# val       575
# test      575

# Label distribution per split
for split in ['train', 'val', 'test']:
    subset = df[df['Split'] == split]
    label_cols = subset.iloc[:, 15:23]
    print(f"\n{split.upper()} label counts:")
    print(label_cols.sum())
```

## Accessing the Full Dataset

### Current Setup: Sample Subset

This repository includes **50-100 representative samples** in `data/samples/` for demo purposes.

**To use:**
```bash
cd data/samples
ls *.npy | wc -l  # Check sample count
```

### Full Dataset Access

**Option 1: Local Storage (if available)**
```python
# If full 6,500 .npy files are stored locally:
DATA_DIR = Path("d:/path/to/full/data")  # Replace with actual path
files = list(DATA_DIR.glob("*.npy"))
print(f"Total files: {len(files)}")  # Should be ~6500
```

**Option 2: Zenodo / Institutional Repository**
- Contact Mamoona Nisar for access to full dataset
- DOI pending (under review)
- Estimated availability: June 2026

**Option 3: Reconstruct from Source**
- Original data from [Jiang et al. (2026)](https://github.com/fletcherjiang/ANA-SelfPacedLearning)
- CLIP embeddings pre-computed on ViT-L/14
- Can request pre-computed embeddings from original authors

## Data Loading in Code

### Using the Data Loader

```python
from src.data_loader import ANAFeatureDataset, load_all_features
from pathlib import Path

# Path to CSV and feature files
csv_path = "data/features_index_english.csv"
features_dir = "data"

# Option 1: Create PyTorch Dataset (recommended)
train_dataset = ANAFeatureDataset(
    csv_path=csv_path,
    features_dir=features_dir,
    split='train',
    normalize_features=True
)
print(f"Train samples: {len(train_dataset)}")
sample = train_dataset[0]
print(f"Features shape: {sample['features'].shape}")  # (257, 768)
print(f"Labels shape: {sample['labels'].shape}")      # (8,)

# Option 2: Load all features at once (memory-intensive)
metadata, cls_embeddings, file_paths = load_all_features(
    data_dir=Path(features_dir),
    max_samples=100  # Limit for demo
)
print(f"Loaded {len(metadata)} samples")
print(f"CLS embeddings shape: {cls_embeddings.shape}")  # (N, 768)
```

### PyTorch DataLoader Integration

```python
from torch.utils.data import DataLoader

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

# Iterate
for batch_idx, batch in enumerate(train_loader):
    features = batch['features']  # [32, 257, 768]
    labels = batch['labels']      # [32, 8]
    print(f"Batch {batch_idx}: features {features.shape}, labels {labels.shape}")
    if batch_idx == 0:
        break  # Stop after first batch for demo
```

## Data Preprocessing

### Features (Embeddings)

- **L2-Normalization:** Applied in `ANAFeatureDataset`
  - `normalized = features / ||features||_2`
  - Ensures consistent scale across samples

- **CLS-Only Variant:** Use only row 0 (CLS token)
  - Reduces memory: 768 dims instead of 257×768
  - Faster training but loses patch-level info

### Labels (Multi-Label)

- **Binary:** Each of 8 classes is 0 or 1
- **Multi-label:** Samples can have 1 or more labels
  - ~35% of samples have ≥2 labels
  - No "background" class
  
- **Handling:** BCEWithLogitsLoss (binary cross-entropy per class)

### Missing Data

- **Rare:** Most samples have complete embeddings
- **Handling:** Skipped during data loading (error handling in data_loader.py)

## Quality Assurance

### Validation Checks

```python
# Verify data integrity
import numpy as np

# 1. Check embedding shapes
embedding = np.load("data/samples/022541326800_20180810B1-2_1：320_peripheralcytoplasm.npy")
assert embedding.shape == (257, 768), f"Expected (257, 768), got {embedding.shape}"

# 2. Check normalization
norm = np.linalg.norm(embedding, axis=1)
assert np.allclose(norm, 1.0, atol=0.01), f"Features not L2-normalized: {norm[:5]}"

# 3. Check label consistency
import pandas as pd
df = pd.read_csv("data/features_index_english.csv")
labels = df.iloc[:, 15:23]
assert labels.isin([0, 1]).all().all(), "Labels not binary"
assert labels.sum(axis=1).min() >= 1, "Sample with no labels found"

print("✓ All validation checks passed")
```

## Citation & Attribution

- **CLIP Embeddings:** Radford et al. (2021), "Learning Transferable Models for Computer Vision Tasks"
- **Original ANA Data:** Jiang et al. (2026), "Self-Paced Learning for Images of Antinuclear Antibodies"
- **Dataset Creator:** Mamoona Nisar, North Dakota State University, 2026

## Support & Questions

- **Data Issues:** Check [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Access Request:** Contact Mamoona Nisar
- **CLIP Details:** See [TECHNICAL.md](TECHNICAL.md#1-architecture)

---

**Last Updated:** May 2026  
**Format Version:** 1.0
