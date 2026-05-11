"""
=============================================================================
PHASE 1-5: Multi-Instance Learning (MIL) Pipeline for ANA Fluorescence
=============================================================================
Phases:
  1. Data Engineering: CSV loader, PyTorch Dataset with train/val/test splits
  2. Token-Aware Architecture: CLS anchor + Medical projection + Class-wise max pooling
  3. Sprint Training Loop: AdamW + BCEWithLogitsLoss + Early stopping on F1-Macro
  4. High-Rigor Evaluation: F1-Macro, mAP, per-class metrics, confusion matrix
  5. Demo Inference: Random test sample prediction + clinical context
=============================================================================
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
from tqdm import tqdm
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix


def _ensure_outdir(outdir: str) -> Path:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def _save_training_history(history: dict, outdir: str) -> None:
    out_path = _ensure_outdir(outdir)
    df = pd.DataFrame(history)
    df.to_csv(out_path / 'mil_training_history.csv', index=False)
    with open(out_path / 'mil_training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

# ============================================================================
# PHASE 1: DATA ENGINEERING (30 Minutes)
# ============================================================================

class ANAFeatureDataset(Dataset):
    """
    PyTorch Dataset for ANA fluorescence multi-label classification.
    Loads pre-computed [257, 768] CLIP features and 8 ICAP binary labels from CSV.
    """
    
    ICAP_CLASSES = [
        'golgi_apparatus',
        'homogeneous',
        'nucleolus',
        'nuclear_dot',
        'centromere',
        'nuclear_membrane_peripheral',
        'cytoplasm_mitochondria',
        'speckled_granular_dfs70'
    ]
    
    def __init__(self, csv_path: str, features_dir: str, split: str = 'train', 
                 normalize_features: bool = True):
        """
        Args:
            csv_path: Path to features_index_english.csv
            features_dir: Path to directory containing .npy files
            split: 'train', 'val', or 'test'
            normalize_features: Whether to L2-normalize embeddings
        """
        self.csv_path = Path(csv_path)
        self.features_dir = Path(features_dir)
        self.split = split
        self.normalize_features = normalize_features
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Filter by split
        self.df = self.df[self.df['Split'] == split].reset_index(drop=True)
        
        print(f"[Phase 1] Loaded {len(self.df)} samples for split '{split}'")
        
        # Extract label columns (columns 16-23, 0-indexed = indices 16:24)
        self.labels_df = self.df.iloc[:, 16:24].values.astype(np.float32)  # 8 classes
        
        print(f"[Phase 1] Label shape: {self.labels_df.shape}")
        print(f"[Phase 1] Label distribution (per-class positive samples):")
        for i, cls_name in enumerate(self.ICAP_CLASSES):
            pos_count = int(self.labels_df[:, i].sum())
            print(f"  - {cls_name}: {pos_count}/{len(self.df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            features: [257, 768] float32 tensor (CLS + patches)
            labels: [8] binary labels (multi-hot)
            metadata: dict with patient/titer info
        """
        row = self.df.iloc[idx]
        feature_file = row['feature_file']
        feature_path = self.features_dir / feature_file
        
        # Load .npy file
        try:
            feats = np.load(feature_path)
            # Handle both 768 and 1024-d (in case of earlier 1024-d exports)
            if feats.shape[1] == 1024:
                features = feats[:, :768].astype(np.float32)  # Take first 768 dims
            else:
                features = feats.astype(np.float32)  # [257, 768]
        except Exception as e:
            print(f"Warning: Failed to load {feature_path}: {e}")
            # Return zeros as fallback (ideally skip this sample in practice)
            features = np.zeros((257, 768), dtype=np.float32)
        
        # Normalize features (L2 normalization for CLIP embeddings)
        if self.normalize_features:
            # Normalize along the embedding dimension (axis=1)
            features = sk_normalize(features, norm='l2', axis=1)
        
        # Get labels
        labels = self.labels_df[idx]  # [8]
        
        # Get metadata
        metadata = {
            'feature_file': feature_file,
            'isbn': row['ISBN'] if pd.notna(row['ISBN']) else 'N/A',
            'titer_level': int(row['titer_7classify_label']) if pd.notna(row['titer_7classify_label']) else 0,
            'three_class_label': int(row['three_classify_label']) if pd.notna(row['three_classify_label']) else 0,
        }
        
        return (
            torch.from_numpy(features),  # [257, 768]
            torch.from_numpy(labels),     # [8]
            metadata
        )


def build_dataloaders(csv_path: str, features_dir: str, batch_size: int = 32, 
                      num_workers: int = 0):
    """
    Builds train/val/test dataloaders with proper splits from CSV.
    """
    print("\n" + "="*70)
    print("PHASE 1: DATA ENGINEERING")
    print("="*70)
    
    datasets = {}
    dataloaders = {}
    
    use_pin_memory = torch.cuda.is_available()

    for split in ['train', 'val', 'test']:
        datasets[split] = ANAFeatureDataset(
            csv_path=csv_path,
            features_dir=features_dir,
            split=split,
            normalize_features=True
        )
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=use_pin_memory
        )
    
    print(f"\n[Phase 1] Dataloaders created:")
    for split in ['train', 'val', 'test']:
        print(f"  - {split}: {len(datasets[split])} samples, "
              f"{len(dataloaders[split])} batches")
    
    return dataloaders, datasets


# ============================================================================
# PHASE 2: TOKEN-AWARE MIL ARCHITECTURE (1 Hour)
# ============================================================================

class MedicalProjectionHead(nn.Module):
    """
    2-layer MLP: 768 → 1024 → 512, with ReLU and Dropout.
    Projects general CLIP features into specialized ANA-fluorescence space.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024, 
                 output_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        """x: [*, 768]"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x  # [*, 512]


class ClassWiseMaxPoolingHead(nn.Module):
    """
    Class-wise max pooling: For each of 8 ICAP classes, identifies the 
    most representative patch token via learned class-specific attention.
    """
    def __init__(self, feature_dim: int = 512, num_classes: int = 8, 
                 num_patches: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_patches = num_patches
        
        # Learnable attention weights per class: [num_classes, feature_dim]
        self.class_weights = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.class_weights)
    
    def forward(self, patch_features):
        """
        patch_features: [batch, 256, 512] (patches only, excluding CLS)
        Returns: [batch, 8, 512]
        """
        batch_size = patch_features.shape[0]
        
        # Compute class-specific attention scores: [batch, 8, 256]
        # scores[b, c, p] = dot(patch_features[b, p], class_weights[c])
        scores = torch.einsum('bpd,cd->bcp', patch_features, self.class_weights)
        
        # Softmax over patches: [batch, 8, 256]
        attention = torch.softmax(scores, dim=2)
        
        # Weighted max per class (approximate max via top-k attention)
        # For each class and batch, take top-3 patches and average
        top_k = min(3, self.num_patches)
        top_attention, top_indices = torch.topk(attention, top_k, dim=2)
        
        # Pool patches: [batch, 8, 512]
        pooled = torch.zeros(batch_size, self.num_classes, self.feature_dim,
                            device=patch_features.device, dtype=patch_features.dtype)
        for b in range(batch_size):
            for c in range(self.num_classes):
                for k in range(top_k):
                    idx = top_indices[b, c, k]
                    pooled[b, c] += top_attention[b, c, k] * patch_features[b, idx]
        
        return pooled  # [batch, 8, 512]


class ANAMILModel(nn.Module):
    """
    Multi-Instance Learning model for ANA pattern classification.
    
    Architecture:
      1. CLS token as global anchor (removed from patches)
      2. Project patches through medical projection head: [256, 768] -> [256, 512]
      3. Class-wise max pooling to identify pattern presence
      4. Fusion: concatenate CLS projection + pooled class features
      5. Output: [8] logits for sigmoid (multi-label)
    """
    def __init__(self, input_dim: int = 768, num_classes: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Medical projection head for both CLS and patches
        self.projection = MedicalProjectionHead(input_dim, 1024, 512)
        
        # Class-wise max pooling for patches
        self.class_pooling = ClassWiseMaxPoolingHead(512, num_classes, 256)
        
        # Final fusion layer: [512 (CLS) + 8*512 (pooled classes)] -> [8]
        fusion_dim = 512 + num_classes * 512
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, features):
        """
        features: [batch, 257, 768]
        Returns: [batch, 8] logits
        """
        # Extract CLS and patches
        cls_token = features[:, 0, :]  # [batch, 768]
        patch_tokens = features[:, 1:, :]  # [batch, 256, 768]
        
        # Project CLS through medical head
        cls_proj = self.projection(cls_token)  # [batch, 512]
        
        # Project patches through medical head
        batch_size = patch_tokens.shape[0]
        patch_tokens_flat = patch_tokens.reshape(-1, self.input_dim)  # [batch*256, 768]
        patch_proj_flat = self.projection(patch_tokens_flat)  # [batch*256, 512]
        patch_proj = patch_proj_flat.reshape(batch_size, 256, 512)  # [batch, 256, 512]
        
        # Class-wise max pooling
        pooled = self.class_pooling(patch_proj)  # [batch, 8, 512]
        pooled_flat = pooled.reshape(batch_size, -1)  # [batch, 8*512]
        
        # Fusion
        fused = torch.cat([cls_proj, pooled_flat], dim=1)  # [batch, 512+8*512]
        logits = self.fusion(fused)  # [batch, 8]
        
        return logits


# ============================================================================
# PHASE 3: SPRINT TRAINING LOOP (2 Hours)
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, use_dspl=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (features, labels, metadata) in enumerate(pbar):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(features)  # [batch, 8]
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate_epoch(model, dataloader, criterion, device, dataset_name='Val'):
    """Evaluate for one epoch. Returns F1-Macro and mAP."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"[{dataset_name}]")
    with torch.no_grad():
        for features, labels, metadata in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_logits = np.concatenate(all_logits, axis=0)  # [N, 8]
    all_labels = np.concatenate(all_labels, axis=0)  # [N, 8]
    
    # Convert logits to probabilities for metrics
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))  # Sigmoid
    all_preds = (all_probs > 0.5).astype(np.int32)
    
    # Compute metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # mAP: per-class average precision, then averaged
    map_scores = []
    for c in range(8):
        try:
            ap = average_precision_score(all_labels[:, c], all_probs[:, c])
            map_scores.append(ap)
        except:
            map_scores.append(0.0)
    mean_ap = np.mean(map_scores)
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'f1_macro': f1_macro,
        'mAP': mean_ap,
        'logits': all_logits,
        'labels': all_labels,
        'probs': all_probs
    }


def train_mil_model(model, dataloaders, datasets, num_epochs: int = 15, 
                    lr: float = 1e-4, weight_decay: float = 0.05, device: str = 'cpu',
                    use_dspl: bool = False):
    """
    Main training loop with early stopping on F1-Macro.
    """
    print("\n" + "="*70)
    print("PHASE 3: SPRINT TRAINING LOOP")
    print("="*70)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = 0.0
    patience = 5
    epochs_no_improve = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1_macro': [],
        'val_mAP': []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, dataloaders['train'], optimizer, criterion, 
                                device, epoch)
        training_history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics = evaluate_epoch(model, dataloaders['val'], criterion, device, 'Val')
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_f1_macro'].append(val_metrics['f1_macro'])
        training_history['val_mAP'].append(val_metrics['mAP'])
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_metrics['loss']:.4f}, "
              f"F1-Macro={val_metrics['f1_macro']:.4f}, "
              f"mAP={val_metrics['mAP']:.4f}")
        
        # Early stopping
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            epochs_no_improve = 0
            print(f"  ✓ F1-Macro improved! Saving checkpoint...")
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  ⚠ Early stopping triggered (no improvement for {patience} epochs)")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    print(f"\nBest F1-Macro: {best_f1:.4f}")
    
    return model, training_history


# ============================================================================
# PHASE 4: HIGH-RIGOR EVALUATION (30 Minutes)
# ============================================================================

def evaluate_model(model, dataloader, device, dataset_name='Test'):
    """
    Comprehensive evaluation with all metrics.
    """
    print("\n" + "="*70)
    print(f"PHASE 4: EVALUATION ({dataset_name})")
    print("="*70)
    
    model.eval()
    all_logits = []
    all_labels = []
    all_metadata = []
    
    with torch.no_grad():
        for features, labels, metadata in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            features = features.to(device)
            logits = model(features)
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_metadata.extend(metadata)
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))
    all_preds = (all_probs > 0.5).astype(np.int32)
    
    # Compute metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # mAP per class
    class_aps = []
    for c in range(8):
        try:
            ap = average_precision_score(all_labels[:, c], all_probs[:, c])
        except:
            ap = 0.0
        class_aps.append(ap)
    
    mean_ap = np.mean(class_aps)
    
    # Print results
    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'F1-Macro':<20} {f1_macro:.4f}")
    print(f"{'F1-Micro':<20} {f1_micro:.4f}")
    print(f"{'F1-Weighted':<20} {f1_weighted:.4f}")
    print(f"{'mAP':<20} {mean_ap:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<35} {'AP':<8} {'Support':<8}")
    print("-" * 55)
    
    class_names = [
        'golgi_apparatus', 'homogeneous', 'nucleolus', 'nuclear_dot',
        'centromere', 'nuclear_membrane_peripheral', 'cytoplasm_mitochondria',
        'speckled_granular_dfs70'
    ]
    
    for c, name in enumerate(class_names):
        support = int(all_labels[:, c].sum())
        print(f"{name:<35} {class_aps[c]:.4f}    {support:<8}")
    
    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'mAP': mean_ap,
        'class_aps': class_aps,
        'logits': all_logits,
        'labels': all_labels,
        'probs': all_probs,
        'preds': all_preds,
        'metadata': all_metadata
    }


# ============================================================================
# PHASE 5: DEMO INFERENCE (Ready for Presentation)
# ============================================================================

def demo_inference(model, dataset, device, num_samples: int = 3):
    """
    Live inference on random test samples with clinical context.
    """
    print("\n" + "="*70)
    print("PHASE 5: DEMO INFERENCE")
    print("="*70)
    
    class_names = [
        'Golgi Apparatus', 'Homogeneous', 'Nucleolus', 'Nuclear Dot',
        'Centromere', 'Nuclear Membrane/Peripheral', 'Cytoplasm/Mitochondria',
        'Speckled/Granular DFS70'
    ]
    
    titer_levels = {1: '100', 2: '320', 3: '1000', 4: '3200', 5: '10000', 6: '32000'}
    
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1}: Index {idx}")
        print(f"{'='*70}")
        
        features, labels, metadata = dataset[idx]
        features_batch = features.unsqueeze(0).to(device)  # [1, 257, 768]
        
        with torch.no_grad():
            logits = model(features_batch)
        
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # [8]
        
        print(f"\nFeature File: {metadata['feature_file']}")
        print(f"Patient (ISBN): {metadata['isbn']}")
        print(f"Titer Level: {titer_levels.get(metadata['titer_level'], 'Unknown')}")
        print(f"Ground Truth Labels: {np.where(labels.numpy())[0]} (class indices)")
        
        print(f"\nPredicted Probabilities (Confidence):")
        print(f"{'Class':<40} {'Probability':<12} {'Predicted'}")
        print("-" * 65)
        
        for c, name in enumerate(class_names):
            prob = probs[c]
            pred = "✓ YES" if prob > 0.5 else "  No"
            print(f"{name:<40} {prob:.4f}        {pred}")
        
        # Top-3 predictions
        top_indices = np.argsort(probs)[::-1][:3]
        print(f"\nTop-3 Predicted Patterns:")
        for rank, c in enumerate(top_indices, 1):
            print(f"  {rank}. {class_names[c]}: {probs[c]:.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Train + evaluate MIL model on pre-extracted CLIP features.')
    parser.add_argument('--csv', default=r'd:\NDSU\research\autoimmune\clip_vit\features\clip_patches_ViT-L_14_with_cls_english\features_index_english.csv')
    parser.add_argument('--features-dir', default=r'd:\NDSU\research\autoimmune\clip_vit\features\clip_patches_ViT-L_14_with_cls_english')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--outdir', default='output')
    args = parser.parse_args()

    # Paths
    CSV_PATH = args.csv
    FEATURES_DIR = args.features_dir
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    OUTDIR = args.outdir
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {DEVICE}")
    print(f"CSV Path: {CSV_PATH}")
    print(f"Features Dir: {FEATURES_DIR}")
    
    # Phase 1: Data Engineering
    dataloaders, datasets = build_dataloaders(CSV_PATH, FEATURES_DIR, BATCH_SIZE)
    
    # Phase 2: Build model
    print("\n" + "="*70)
    print("PHASE 2: TOKEN-AWARE MODEL ARCHITECTURE")
    print("="*70)
    model = ANAMILModel(input_dim=768, num_classes=8)
    print(f"Model built successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Phase 3: Train
    model, history = train_mil_model(model, dataloaders, datasets, num_epochs=NUM_EPOCHS,
                                    lr=1e-4, weight_decay=0.05, device=DEVICE, use_dspl=False)

    # Save training curves
    _save_training_history(history, OUTDIR)
    with open(_ensure_outdir(OUTDIR) / 'mil_run_info.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'device': DEVICE,
                'csv_path': str(CSV_PATH),
                'features_dir': str(FEATURES_DIR),
                'batch_size': BATCH_SIZE,
                'num_epochs_requested': NUM_EPOCHS,
                'checkpoint': 'best_model.pt',
            },
            f,
            indent=2,
        )
    
    # Phase 4: Evaluate on test
    test_metrics = evaluate_model(model, dataloaders['test'], DEVICE, 'Test')

    # Save headline metrics (keeps file small; evaluation script can recompute curves/CM)
    with open(_ensure_outdir(OUTDIR) / 'mil_test_headline_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'f1_macro': float(test_metrics['f1_macro']),
                'f1_micro': float(test_metrics['f1_micro']),
                'f1_weighted': float(test_metrics['f1_weighted']),
                'mAP': float(test_metrics['mAP']),
                'class_aps': [float(x) for x in test_metrics['class_aps']],
                'classes': ANAFeatureDataset.ICAP_CLASSES,
            },
            f,
            indent=2,
        )
    
    # Phase 5: Demo inference
    demo_inference(model, datasets['test'], DEVICE, num_samples=3)
    
    print("\n" + "="*70)
    print("✅ MIL PIPELINE COMPLETE")
    print("="*70)
