# Data Directory

CLIP embeddings and index files for ANA pattern classification.

## Files

### features_index_english.csv
CSV index mapping samples to embeddings and labels.

**Structure:**
- ~4,000 rows (indexed samples)
- 23 columns
- Key columns:
  - Col 2 (idx=1): Embedding filename (.npy path)
  - Col 9 (idx=8): Split (train/val/test)
  - Cols 16-23 (idx=15:23): Binary labels for 8 ICAP classes

**Labels (Columns 15-22):**
1. golgi_apparatus
2. homogeneous
3. nucleolus
4. nuclear_dot
5. centromere
6. nuclear_membrane_peripheral
7. cytoplasm_mitochondria
8. speckled_granular_dfs70

See [Data Documentation](../docs/DATA.md) for full details.

### *.npy (Embedding Files)
Individual CLIP ViT-L/14 embeddings:
- **Shape:** (257, 768)
- **Type:** float32 numpy array
- **Content:** 1 CLS token + 256 patch tokens
- **Normalization:** L2-normalized

Example filename: `022541326800_20180810B1-2_1：320_peripheralcytoplasm.npy`

## Accessing Full Dataset

### Current: Sample Subset
This repo includes ~75 sample .npy files in `samples/` for demonstration.

### Full Dataset (~6,500 files, ~100GB)
- Contact Mamoona Nisar for access
- Alternative: Request from original Jiang et al. authors
- Will be available on Zenodo (DOI pending)

## Loading Data

```python
from src.data_loader import ANAFeatureDataset

# Create dataset
dataset = ANAFeatureDataset(
    csv_path="data/features_index_english.csv",
    features_dir="data",
    split='train',
    normalize_features=True
)

# Get sample
sample = dataset[0]
print(sample['features'].shape)  # [257, 768]
print(sample['labels'].shape)    # [8]
```

See [Data Documentation](../docs/DATA.md) for detailed guidance.

---

**Questions?** Check [Data Documentation](../docs/DATA.md) or open an issue.
