# Inference Guide: Using the Pretrained Model

## Quick Start

```python
import torch
import numpy as np
from src.mil_pipeline import ANAMILModel

# Load model
model = ANAMILModel()
checkpoint = torch.load("models/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load embedding: [257, 768]
embedding = np.load("data/samples/sample_embedding.npy")
embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

# Inference
with torch.no_grad():
    logits = model(embedding_tensor)      # [1, 8] raw scores
    probabilities = torch.sigmoid(logits)  # [1, 8] probabilities (0-1)
    predictions = (probabilities >= 0.5).int()  # [1, 8] binary predictions

# Interpret results
class_names = [
    'golgi_apparatus', 'homogeneous', 'nucleolus', 'nuclear_dot',
    'centromere', 'nuclear_membrane_peripheral', 'cytoplasm_mitochondria',
    'speckled_granular_dfs70'
]

print("ANA Pattern Predictions:")
for i, cls_name in enumerate(class_names):
    pred = predictions[0, i].item()
    prob = probabilities[0, i].item()
    print(f"  {cls_name}: {pred} (confidence: {prob:.2%})")
```

## Model Architecture

### Input/Output

- **Input:** [batch_size, 257, 768] tensor (CLIP embeddings)
  - 1 CLS token + 256 patch tokens
  - float32, L2-normalized

- **Output:** [batch_size, 8] logits
  - 8 classes (ICAP ANA patterns)
  - Apply sigmoid for probabilities: `P = 1 / (1 + exp(-logits))`
  - Apply threshold for binary predictions: `pred = P >= 0.5`

### Model Components

```python
model = ANAMILModel(
    input_dim=768,      # CLIP embedding dimension
    hidden_dim=1024,    # Projection head hidden layer
    output_dim=512,     # Projection head output dimension
    num_classes=8       # Number of ANA patterns
)

# Components:
# 1. CLS extraction (index 0 from [257, 768])
# 2. Projection head: 768 → 1024 → 512
# 3. Class-specific max pooling
# 4. 8 sigmoid outputs
```

## Loading Pretrained Weights

### Option 1: Load Checkpoint

```python
import torch
from src.mil_pipeline import ANAMILModel

# Initialize model
model = ANAMILModel()

# Load checkpoint
checkpoint = torch.load("models/best_model.pt", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded successfully")
print(f"  Test F1-Macro: {checkpoint['metrics']['f1_macro']:.4f}")
print(f"  Test mAP: {checkpoint['metrics']['map']:.4f}")
```

### Option 2: Device Handling (GPU/CPU)

```python
import torch
from src.mil_pipeline import ANAMILModel

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = ANAMILModel().to(device)
checkpoint = torch.load("models/best_model.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Single Sample Inference

### Load Embedding

```python
import numpy as np
import torch

# Load .npy file
embedding = np.load("data/samples/sample.npy")
print(f"Embedding shape: {embedding.shape}")  # Should be (257, 768)

# Convert to tensor
embedding_tensor = torch.tensor(embedding, dtype=torch.float32)

# Add batch dimension
batch = embedding_tensor.unsqueeze(0)  # [1, 257, 768]
print(f"Batch shape: {batch.shape}")
```

### Get Predictions

```python
with torch.no_grad():
    # Forward pass
    logits = model(batch.to(device))  # [1, 8]
    
    # Convert to probabilities
    probabilities = torch.sigmoid(logits)  # [1, 8] in range [0, 1]
    
    # Binary predictions (threshold = 0.5)
    predictions = (probabilities >= 0.5).int()  # [1, 8]

# Extract to CPU/numpy
probs = probabilities.cpu().numpy()[0]  # [8]
preds = predictions.cpu().numpy()[0]    # [8]

print(f"Probabilities: {probs}")
print(f"Predictions: {preds}")
```

### Interpret Multi-Label Output

```python
class_names = [
    'golgi_apparatus',
    'homogeneous',
    'nucleolus',
    'nuclear_dot',
    'centromere',
    'nuclear_membrane_peripheral',
    'cytoplasm_mitochondria',
    'speckled_granular_dfs70'
]

print("=" * 60)
print("ANA PATTERN CLASSIFICATION RESULTS")
print("=" * 60)

detected_patterns = []
for i, cls_name in enumerate(class_names):
    prob = probs[i]
    pred = preds[i]
    
    if pred == 1:
        detected_patterns.append(cls_name)
        status = "✓ DETECTED"
    else:
        status = "✗ Not detected"
    
    print(f"{cls_name:35s} {prob:6.1%}  {status}")

print("=" * 60)
if detected_patterns:
    print(f"Detected patterns: {', '.join(detected_patterns)}")
else:
    print("No patterns detected (all probabilities < 50%)")
print("=" * 60)
```

## Batch Inference

### Process Multiple Samples

```python
import numpy as np
import torch
from pathlib import Path

# Get list of embedding files
data_dir = Path("data/samples")
embedding_files = list(data_dir.glob("*.npy"))[:10]  # First 10 for demo

# Load all embeddings
embeddings = []
for f in embedding_files:
    emb = np.load(f)
    embeddings.append(emb)

# Stack into batch
batch = torch.tensor(np.array(embeddings), dtype=torch.float32).to(device)
print(f"Batch shape: {batch.shape}")  # [N, 257, 768]

# Batch inference
with torch.no_grad():
    logits = model(batch)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).int()

# Extract to numpy
probs = probabilities.cpu().numpy()  # [N, 8]
preds = predictions.cpu().numpy()    # [N, 8]

# Display results
for i, file in enumerate(embedding_files):
    detected = [class_names[j] for j in range(8) if preds[i, j] == 1]
    print(f"{file.name}: {', '.join(detected) if detected else 'None'}")
```

### Parallel Batch Processing

```python
import torch
from torch.utils.data import DataLoader

# For large-scale inference, use DataLoader
test_dataset = ANAFeatureDataset(
    csv_path="data/features_index_english.csv",
    features_dir="data",
    split='test'
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,  # Process 64 samples at once
    shuffle=False,
    num_workers=4
)

all_predictions = []
all_probabilities = []

model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        features = batch['features'].to(device)  # [64, 257, 768]
        
        # Forward pass
        logits = model(features)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()
        
        # Store results
        all_probabilities.append(probs.cpu())
        all_predictions.append(preds.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * 64} samples")

# Concatenate all results
all_probs = torch.cat(all_probabilities).numpy()  # [total, 8]
all_preds = torch.cat(all_predictions).numpy()    # [total, 8]

print(f"Inference complete: {all_probs.shape[0]} samples processed")
```

## Confidence Thresholds

### Per-Class Thresholds

**Problem:** Default threshold (0.5) may not be optimal for all classes.

**Solution:** Adjust per-class thresholds based on validation set:

```python
# Find optimal threshold per class
thresholds = np.zeros(8)

val_probs = ...  # [N_val, 8] validation probabilities
val_labels = ... # [N_val, 8] validation labels

from sklearn.metrics import precision_recall_curve

for c in range(8):
    precision, recall, threshs = precision_recall_curve(val_labels[:, c], val_probs[:, c])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    thresholds[c] = threshs[best_idx]

print(f"Optimal thresholds: {thresholds}")

# Use per-class thresholds
predictions_optimized = (val_probs >= thresholds).astype(int)
```

## Probabilistic Calibration

### Issue: Over-Confident Predictions

For some classes (especially minority classes), model may be over-confident or under-confident.

### Solution: Temperature Scaling

```python
from scipy.optimize import minimize

def temperature_scale(logits, labels, T_init=1.0):
    """Find optimal temperature for probability calibration."""
    def neg_log_likelihood(T):
        probs = torch.sigmoid(logits / T)
        loss = torch.nn.BCELoss()(probs, labels)
        return loss.item()
    
    result = minimize(neg_log_likelihood, T_init, method='Nelder-Mead')
    return result.x[0]

# Find temperature on validation set
val_logits = ...  # [N_val, 8]
val_labels = ...  # [N_val, 8]

T = temperature_scale(val_logits, val_labels)
print(f"Optimal temperature: {T:.3f}")

# Apply during inference
calibrated_probs = torch.sigmoid(logits / T)
```

## Clinical Interpretation

### Understanding ANA Patterns

| Pattern | Clinical Significance | Fluorescence Appearance |
|---------|----------------------|------------------------|
| **cytoplasm_mitochondria** | Mitochondrial antibodies (anti-Jo1, anti-Ro) | Cytoplasmic dots |
| **nuclear_membrane_peripheral** | Peripheral nuclear envelope antibodies | Ring around nucleus |
| **nucleolus** | Anti-fibrillarin, anti-RNA polymerase | Nucleolus staining |
| **speckled_granular_dfs70** | DFS70 antibodies (IIF ELISA negative) | Fine speckled pattern |
| **homogeneous** | Anti-dsDNA, anti-histone | Uniform nuclear staining |
| **centromere** | Anti-centromere (scleroderma) | Discrete dots at centromeres |
| **nuclear_dot** | Fine dots in nucleus | Small punctate pattern |
| **golgi_apparatus** | Anti-Golgi, anti-giantin | Perinuclear Golgi pattern |

### Multi-Label Interpretation

Many samples have **multiple simultaneous patterns**:

```python
# Example: cytoplasm_mitochondria + nuclear_membrane_peripheral
detected = ['cytoplasm_mitochondria', 'nuclear_membrane_peripheral']
clinical_note = (
    "Multiple antibodies detected:\n"
    "- Mitochondrial antibodies (anti-Jo1/anti-Ro)\n"
    "- Nuclear envelope antibodies\n"
    "Consider: Overlap syndrome, myositis"
)
```

## Error Handling

### Robust Inference

```python
import numpy as np
import torch
from pathlib import Path

def safe_inference(embedding_path, model, device, class_names):
    """
    Safely load embedding and perform inference with error handling.
    """
    try:
        # Load embedding
        if not Path(embedding_path).exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        embedding = np.load(embedding_path)
        
        # Validate shape
        if embedding.shape != (257, 768):
            raise ValueError(f"Invalid shape: expected (257, 768), got {embedding.shape}")
        
        # Convert to tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(embedding_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
        
        # Return results
        return {
            'success': True,
            'probabilities': probs[0].cpu().numpy(),
            'predictions': preds[0].cpu().numpy(),
            'class_names': class_names
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Usage
result = safe_inference("data/samples/sample.npy", model, device, class_names)
if result['success']:
    print("Inference successful")
    for i, cls in enumerate(result['class_names']):
        print(f"  {cls}: {result['probabilities'][i]:.2%}")
else:
    print(f"Inference failed: {result['error']}")
```

## Performance Notes

### Inference Speed

- **Single Sample:** ~5 ms (CPU), ~1 ms (GPU)
- **Batch of 64:** ~50 ms (CPU), ~10 ms (GPU)
- **Bottleneck:** Data loading (.npy reading), not model inference

### Memory Usage

- **Model Size:** ~5 MB (weights only)
- **Per Sample:** 0.8 MB (257×768 embeddings)
- **Batch of 64:** ~52 MB (embeddings only)

## Troubleshooting

### Issue: `RuntimeError: Input has NaN values`
```python
# Check for NaNs in embedding
if np.isnan(embedding).any():
    print("Embedding contains NaN values")
    # Handle or skip this sample
```

### Issue: `shape mismatch: expected (257, 768)`
```python
# Ensure correct shape
embedding = np.load(path)
if embedding.shape != (257, 768):
    print(f"Shape mismatch: {embedding.shape}")
    # Maybe it's transposed?
    if embedding.shape == (768, 257):
        embedding = embedding.T
```

### Issue: Predictions always 0 (no patterns detected)
```python
# Check probability distribution
with torch.no_grad():
    logits = model(batch)
    probs = torch.sigmoid(logits)

print("Probability statistics:")
print(f"  Mean: {probs.mean():.3f}")
print(f"  Min: {probs.min():.3f}, Max: {probs.max():.3f}")

# If all < 0.5, model is under-confident
# Solution: Lower threshold or recalibrate
```

---

**Last Updated:** May 2026  
**Model:** best_model.pt (F1-Macro: 0.1522, mAP: 0.3147)
