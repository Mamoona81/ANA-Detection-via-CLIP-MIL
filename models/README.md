# Models Directory

Pretrained model checkpoints for ANA pattern classification.

## best_model.pt

**Trained Model Checkpoint**
- **Architecture:** ANAMILModel (CLS-anchor + projection head)
- **Training Date:** May 2026
- **Performance:**
  - F1-Macro: **0.1522**
  - F1-Micro: 0.5447
  - mAP: 0.3147
- **Size:** ~5 MB
- **Format:** PyTorch state_dict (.pt binary)

### Loading

```python
import torch
from src.mil_pipeline import ANAMILModel

model = ANAMILModel()
checkpoint = torch.load("models/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': {...},  # Model weights
    'optimizer_state_dict': {...},
    'epoch': 45,
    'metrics': {
        'f1_macro': 0.1522,
        'f1_micro': 0.5447,
        'map': 0.3147,
        'per_class_ap': [...]
    },
    'training_history': {...}
}
```

## Future Models

Additional checkpoints may be added for:
- Focal loss variant (improved minority class performance)
- Calibrated model (temperature scaling for confidence)
- Attention-based architecture (interpretability)

---

See [Inference Guide](../docs/INFERENCE.md) for usage examples.
