# Sample Data

This directory contains representative CLIP embedding samples for testing and demonstration.

## Contents

- ~75 sample .npy files (~300 MB total)
- Covers multiple ANA patterns
- Balanced representation across classes

## Usage

```python
import numpy as np
from pathlib import Path

# Load a sample
sample_dir = Path("data/samples")
sample_file = sample_dir / "022541326800_20180810B1-2_1：320_peripheralcytoplasm.npy"

embedding = np.load(sample_file)
print(embedding.shape)  # (257, 768)
```

## Full Dataset

For the complete ~6,500 embedding files, please:
1. Contact Mamoona Nisar
2. Request from original Jiang et al. authors
3. Check Zenodo repository (DOI pending)

---

See [Data Documentation](../DATA.md) for access instructions.
