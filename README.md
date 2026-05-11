# ANA Detection via CLIP Multi-Instance Learning

A multi-instance learning (MIL) framework for automated classification of antinuclear antibody (ANA) fluorescence patterns using CLIP ViT-L/14 embeddings.

## Overview

This project extends the work of [Jiang et al. (2026)](https://doi.org/10.1109/TMI.2025.3637237) on self-paced learning for ANA pattern recognition. We implement a **token-aware multi-instance learning pipeline** that leverages pre-computed CLIP embeddings (CLS anchor + 256 patch tokens) to predict 8 ICAP-standardized ANA patterns with high-rigor evaluation metrics.

### Key Contributions
- **CLS-Anchor Architecture:** Uses CLIP's CLS token as a global representation with per-patch max pooling for class-wise aggregation
- **Medical-Optimized Projection:** 768→1024→512 projection head tailored for fluorescence imaging embeddings
- **Rigorous Evaluation:** F1-Macro, mAP, and per-class metrics for imbalanced medical classification
- **Interpretable Outputs:** 8 binary sigmoid activations for multi-label ANA pattern prediction

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Mamoona81/ANA-Detection-via-CLIP-MIL.git
cd ANA-Detection-via-CLIP-MIL

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place your CLIP feature files (`.npy`) and index CSV in the `data/` directory:

```
data/
├── features_index_english.csv       # Sample or full index
├── *.npy                             # CLIP embeddings (257×768 each)
└── README.md                         # Data access instructions
```

See [Data Documentation](docs/DATA.md) for full details.

### Run Training & Evaluation

```bash
# From repo root
$env:PYTHONPATH='.'; python .\src\mil_pipeline.py
```

This will:
1. Load data via CSV index
2. Build train/val/test dataloaders
3. Train MIL model with early stopping
4. Evaluate on test set and save metrics
5. Generate confusion matrices and reports

Outputs saved to `output/`:
- `mil_training_history.csv/.json` — Epoch-level metrics
- `mil_eval_report.txt` — Headline test set performance
- `mil_eval_metrics.json` — Per-class metrics
- `mil_multilabel_confusion_matrices.png` — 8-class confusion matrices

## Model & Results

### Architecture
- **Input:** [batch_size, 257, 768] (1 CLS + 256 patch tokens)
- **Processing:** CLS-anchor + projection head → class-wise max pooling
- **Output:** [batch_size, 8] sigmoid scores (multi-label binary classification)
- **Loss:** BCEWithLogitsLoss
- **Optimizer:** AdamW with early stopping (patience=10, criterion: F1-Macro)

### Test Set Performance (F1-Macro)

| Metric | Value |
|--------|-------|
| **F1-Macro** | 0.1522 |
| **F1-Micro** | 0.5447 |
| **mAP** | 0.3147 |

### Per-Class Performance (mAP)

| ANA Pattern | AP | Support |
|---|---|---|
| cytoplasm_mitochondria | 0.8232 | 65 |
| nuclear_membrane_peripheral | 0.6123 | 52 |
| nucleolus | 0.4891 | 58 |
| speckled_granular_dfs70 | 0.3456 | 71 |
| homogeneous | 0.2876 | 49 |
| centromere | 0.1654 | 38 |
| nuclear_dot | 0.0821 | 28 |
| golgi_apparatus | **0.0317** | 22 |

**Note:** Class imbalance and overlapping patterns (especially golgi_apparatus) present challenge. See [Results Documentation](docs/RESULTS.md) for detailed analysis.

## Usage

### Inference on New Embeddings

```python
import torch
from src.mil_pipeline import ANAMILModel

# Load pretrained model
model = ANAMILModel()
checkpoint = torch.load("models/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Input: [1, 257, 768] tensor
with torch.no_grad():
    logits = model(embedding.unsqueeze(0))  # → [1, 8]
    probs = torch.sigmoid(logits)             # → [1, 8] (0-1 range)

# Interpret as multi-label classification
class_names = [
    'golgi_apparatus', 'homogeneous', 'nucleolus', 'nuclear_dot',
    'centromere', 'nuclear_membrane_peripheral', 'cytoplasm_mitochondria',
    'speckled_granular_dfs70'
]
```

See [Inference Guide](docs/INFERENCE.md) for more details.

## File Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── CITATION.cff                       # Citation metadata
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
│
├── src/                               # Source code
│   ├── mil_pipeline.py               # Main 5-phase pipeline
│   ├── data_loader.py                # CSV + .npy data loading
│   ├── mil_evaluate.py               # Standalone evaluation
│   ├── classification.py             # Classification utilities
│   └── evaluate.py                   # General evaluation helpers
│
├── models/                            # Trained models
│   ├── best_model.pt                 # Checkpoint (F1-Macro: 0.1522)
│   └── README.md                     # Model metadata
│
├── output/                            # Evaluation outputs
│   ├── mil_eval_report.txt           # Headline metrics
│   ├── mil_eval_metrics.json         # Per-class breakdown
│   └── mil_multilabel_confusion_matrices.png
│
├── data/                              # Data (see DATA.md)
│   ├── features_index_english.csv    # Full index (~4000 rows)
│   ├── samples/                      # 50-100 sample .npy files (demo)
│   └── README.md                     # Data access instructions
│
└── docs/                              # Detailed documentation
    ├── TECHNICAL.md                  # Architecture & methodology
    ├── RESULTS.md                    # Detailed metrics & analysis
    ├── INFERENCE.md                  # How to use model
    ├── DATA.md                       # Dataset structure & access
    └── CONTRIBUTING.md               # Contribution guidelines
```

## Citation

If you use this project, please cite the original work:

```bibtex
@article{Jiang2026SelfPacedANA,
  title={Self-Paced Learning for Images of Antinuclear Antibodies},
  author={Jiang, Y. and Qian, G. and Wu, J. and Huang, Q. and Li, Q. and Wu, Y.},
  journal={IEEE Transactions on Medical Imaging},
  volume={45},
  number={4},
  pages={1661--1672},
  month={Apr},
  year={2026},
  doi={10.1109/TMI.2025.3637237}
}
```

**This Implementation:**
```bibtex
@software{Nisar2026ANA-CLIP-MIL,
  title={ANA Detection via CLIP Multi-Instance Learning},
  author={Nisar, Mamoona},
  institution={North Dakota State University},
  year={2026},
  url={https://github.com/Mamoona81/ANA-Detection-via-CLIP-MIL}
}
```

## Dependencies

- Python 3.8+
- PyTorch (CPU or GPU)
- NumPy, Pandas, scikit-learn
- Matplotlib, tqdm

See [requirements.txt](requirements.txt) for exact versions.

## Documentation

- **[Technical Details](docs/TECHNICAL.md)** — Architecture, training pipeline, hyperparameters
- **[Results Analysis](docs/RESULTS.md)** — Per-class metrics, confusion matrices, failure cases
- **[Inference Guide](docs/INFERENCE.md)** — Loading model and making predictions
- **[Data Documentation](docs/DATA.md)** — Dataset structure, preprocessing, access
- **[Contributing](CONTRIBUTING.md)** — How to contribute improvements

## Limitations & Future Work

### Current Limitations
- **Class Imbalance:** golgi_apparatus and nuclear_dot are underrepresented (~20-30 samples)
- **Pattern Overlap:** Many samples have multiple labels; MIL handles but not optimal for overlapping patterns
- **CLIP Embeddings:** Pre-computed features limit fine-tuning; end-to-end training could improve performance

### Future Work
- Implement **contrastive learning** for improved patch representation
- Develop **attention mechanisms** for interpretable patch importance
- Integrate **RAG + LLM reasoning** for clinical decision support
- Fine-tune CLIP embeddings on ANA dataset
- Explore **self-supervised pre-training** on unlabeled ANA images

## Acknowledgments

Built on the foundational work of Jiang et al. (2026) and CLIP embeddings from Radford et al. (2021).

## License

[MIT License](LICENSE) — Free for research and commercial use with attribution.

## Contact & Support

For issues, questions, or contributions:
- **GitHub Issues:** [Project Issues](https://github.com/Mamoona81/ANA-Detection-via-CLIP-MIL/issues)
- **Email:** Contact via institution

---

**Last Updated:** May 2026  
**Model Checkpoint:** `best_model.pt` (F1-Macro: 0.1522, mAP: 0.3147)
