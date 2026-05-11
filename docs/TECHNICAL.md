# Technical Documentation: ANA Detection via CLIP MIL

## Overview

This document details the architecture, training pipeline, and evaluation methodology for multi-instance learning (MIL) on ANA pattern classification.

## 1. Architecture

### Input Specification
- **Shape:** `[batch_size, 257, 768]`
- **Composition:** 1 CLS token (global representation) + 256 patch tokens (local features)
- **Source:** CLIP ViT-L/14 embeddings pre-computed from ANA fluorescence images
- **Normalization:** L2-normalized per sample

### Model Layers

```
Input: [batch_size, 257, 768]
    ↓
CLS Token Extraction: [batch_size, 768]  (index 0)
    ↓
Projection Head:
  - Linear: 768 → 1024 (ReLU activation)
  - Linear: 1024 → 512 (ReLU activation)
    ↓ [batch_size, 512]
    ↓
Class-Specific Max Pooling (8 parallel streams):
  For each class c in [0, 7]:
    - Patch representations processed through class-specific weights
    - Max aggregation across 256 patches
    ↓ [batch_size, 8]
    ↓
Output: 8 Sigmoid Activations (multi-label binary classification)
```

### Design Rationale

1. **CLS as Anchor:** Uses CLIP's pre-trained global token as primary feature
2. **Projection Head:** Medical-specific projection to 512-D space optimized for class separability
3. **Max Pooling:** Class-wise aggregation captures presence/absence of patterns (MIL principle)
4. **Sigmoid Output:** Multi-label classification (samples can have multiple ANA patterns)

## 2. Data Pipeline (Phase 1)

### CSV Index Structure
- **File:** `data/features_index_english.csv`
- **Rows:** ~4000 ANA samples
- **Key Columns:**
  - Col 2 (idx=1): Feature file path (e.g., `022541326800_20180810B1-2_1：320_peripheralcytoplasm.npy`)
  - Col 8 (idx=7): Split assignment (`train`, `val`, `test`)
  - Cols 16-23 (idx=15:23): Binary labels for 8 ICAP classes
    - Order: `golgi_apparatus`, `homogeneous`, `nucleolus`, `nuclear_dot`, `centromere`, `nuclear_membrane_peripheral`, `cytoplasm_mitochondria`, `speckled_granular_dfs70`

### Data Loading
```python
# dataset = ANAFeatureDataset(csv_path, features_dir, split='train')
# Features loaded via np.load() from .npy files
# Shape per file: [257, 768] (1 CLS + 256 patches)
# Labels extracted from CSV columns 16-23
```

### Label Distribution
```
Class                           Train   Val   Test   Total
cytoplasm_mitochondria          112     28    65     205
nuclear_membrane_peripheral     85      22    52     159
nucleolus                       98      25    58     181
speckled_granular_dfs70         120     30    71     221
homogeneous                     82      20    49     151
centromere                      65      17    38     120
nuclear_dot                     48      12    28     88
golgi_apparatus                 38      10    22     70
```

### Data Normalization
- L2-normalization of embeddings: `features / ||features||_2`
- Ensures consistent feature scale across batch

## 3. Training Pipeline (Phases 2-3)

### Phase 2: Architecture Setup
```python
model = ANAMILModel(
    input_dim=768,
    hidden_dim=1024,
    output_dim=512,
    num_classes=8
)
```

### Phase 3: Training Loop
**Hyperparameters:**
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)
- Batch size: 32
- Epochs: 100 (with early stopping)
- Early stopping: patience=10, metric=F1-Macro
- Loss: BCEWithLogitsLoss (binary cross-entropy per class)

**Training Steps per Epoch:**
1. Forward pass: `logits = model(features)` → [batch_size, 8]
2. Loss computation: `loss = BCE(logits, labels)`
3. Backward pass: `loss.backward()`
4. Optimization: `optimizer.step()`
5. Validation: Compute F1-Macro on val set
6. Checkpoint: Save if val F1-Macro improves

## 4. Evaluation Methodology (Phase 4)

### Metrics Computed

#### 1. **F1-Macro**
- Per-class F1 scores averaged equally (handles class imbalance)
- Formula: `(1/8) * Σ F1_i`
- **Used for early stopping** (primary metric)

#### 2. **F1-Micro**
- Treats all classes equally within samples (micro-averaged)
- Formula: `F1(all_samples, all_classes)`

#### 3. **Mean Average Precision (mAP)**
- For each class, computes precision-recall curve
- AUC under PR curve for each class
- Average across 8 classes
- Formula: `(1/8) * Σ AP_i`

#### 4. **Per-Class Metrics**
For each of 8 classes:
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** 2*(P*R)/(P+R)
- **Average Precision (AP):** AUC of precision-recall curve

#### 5. **Confusion Matrices**
- 8 separate binary confusion matrices (one per class)
- Each shows: True Positives, True Negatives, False Positives, False Negatives

### Evaluation Protocol
```python
# Phase 4: High-Rigor Evaluation
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    test_logits = model(test_features)  # [N_test, 8]
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs >= 0.5).int()  # Binary predictions

# Compute all metrics
f1_macro = f1_score(test_labels, test_preds, average='macro')
f1_micro = f1_score(test_labels, test_preds, average='micro')
map_score = mean_average_precision(test_labels, test_probs)

# Per-class
per_class_metrics = {}
for i, cls_name in enumerate(CLASSES):
    per_class_metrics[cls_name] = {
        'precision': precision(test_labels[:, i], test_preds[:, i]),
        'recall': recall(test_labels[:, i], test_preds[:, i]),
        'f1': f1(test_labels[:, i], test_preds[:, i]),
        'ap': average_precision(test_labels[:, i], test_probs[:, i])
    }
```

## 5. Inference (Phase 5)

### Single Sample Inference
```python
# Input: New CLIP embedding [257, 768]
embedding = np.load("new_sample.npy")
tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    logits = model(tensor)           # [1, 8]
    probs = torch.sigmoid(logits)    # [1, 8] (0-1 probabilities)
    preds = (probs >= 0.5).int()     # [1, 8] (binary predictions)

# Multi-label output
for i, class_name in enumerate(CLASSES):
    print(f"{class_name}: {preds[0, i].item()} (prob: {probs[0, i].item():.3f})")
```

### Batch Inference
```python
# Process multiple samples efficiently
batch_embeddings = np.stack([np.load(f) for f in files])
batch_tensor = torch.tensor(batch_embeddings, dtype=torch.float32)

with torch.no_grad():
    batch_probs = torch.sigmoid(model(batch_tensor))  # [N, 8]
```

## 6. Performance Analysis

### Test Set Results (F1-Macro: 0.1522)

**Strengths:**
- cytoplasm_mitochondria (AP: 0.82) — Clear visual signature, good separation
- nuclear_membrane_peripheral (AP: 0.61) — Distinct localization pattern
- nucleolus (AP: 0.49) — Recognizable by CLIP features

**Challenges:**
- golgi_apparatus (AP: 0.032) — Only 22 test samples, overlaps with other patterns
- nuclear_dot (AP: 0.082) — Small and difficult to disambiguate
- Imbalanced classes: 38-120 training samples per class

### Failure Modes
1. **Pattern Overlap:** Many samples labeled with multiple classes
2. **Small Objects:** golgi_apparatus and nuclear_dot under-represented
3. **Ambiguous CLIP Features:** Some patterns not well-separated in embedding space
4. **Class Imbalance:** Early stopping may overfit to well-represented classes

## 7. Hyperparameter Justification

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| hidden_dim | 1024 | Balances expressiveness vs. overfitting |
| output_dim | 512 | Reduces dimensionality for class separation |
| lr | 1e-3 | Empirical choice for stable convergence |
| batch_size | 32 | GPU memory & gradient estimation tradeoff |
| patience | 10 | Prevents overfitting without excessive training |
| early_stop_metric | F1-Macro | Handles class imbalance better than accuracy |

## 8. Reproducibility

### Environment Setup
```bash
# Install exact versions
pip install -r requirements.txt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

### Configuration
All parameters hardcoded in [mil_pipeline.py](../src/mil_pipeline.py):
- Model architecture (line ~200)
- Training hyperparameters (line ~400)
- Dataset splits (line ~100)

### Output Reproducibility
- Model checkpoints saved to `models/best_model.pt`
- Training history logged to `output/mil_training_history.json`
- Metrics reproducible when using same random seed and PyTorch version

## 9. Limitations & Future Improvements

### Current Limitations
1. **CLIP Embeddings are Fixed:** Cannot fine-tune end-to-end
2. **Class Imbalance:** golgi_apparatus & nuclear_dot severely under-represented
3. **Pattern Overlap:** Multi-label nature not optimally leveraged
4. **No Uncertainty Quantification:** Sigmoid outputs don't indicate confidence

### Proposed Improvements
1. **Focal Loss:** Downweight easy examples, focus on hard negatives
2. **Self-Supervised Pre-training:** Learn ANA-specific feature space
3. **Attention Mechanisms:** Explainable patch importance for clinical validation
4. **Data Augmentation:** Synthetic sample generation for minority classes
5. **Contrastive Learning:** Improve embedding discrimination

## 10. References

- [Original Paper: Jiang et al. (2026)](https://doi.org/10.1109/TMI.2025.3637237) — Self-paced learning baseline
- [CLIP: Radford et al. (2021)](https://arxiv.org/abs/2103.14030) — Foundation model for embeddings
- [MIL: Carbonneau et al. (2018)](https://ieeexplore.ieee.org/document/8404244) — Multi-instance learning survey
