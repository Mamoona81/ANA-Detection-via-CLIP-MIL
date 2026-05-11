r"""mil_evaluate.py

Separate evaluation/visualization script for the MIL pipeline.

Outputs (in outdir, default: output/):
  - mil_training_curves.png
  - mil_multilabel_confusion_matrices.png
  - mil_eval_report.txt
  - mil_eval_metrics.json

Usage:
    $env:PYTHONPATH='.'; python .\\src\\mil_evaluate.py

Notes:
  - This script does NOT retrain.
  - It loads `best_model.pt` and evaluates on the CSV-defined test split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score,
    average_precision_score,
    multilabel_confusion_matrix,
)

from src.mil_pipeline import ANAFeatureDataset, ANAMILModel

# Relative paths (from repo root)
DEFAULT_CSV = str(Path(__file__).parent.parent / "data" / "features_index_english.csv")
DEFAULT_FEATURES_DIR = str(Path(__file__).parent.parent / "data")


def _ensure_outdir(outdir: str) -> Path:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_training_history(history_csv: Path) -> Optional[Dict[str, np.ndarray]]:
    if not history_csv.exists():
        return None

    import pandas as pd

    df = pd.read_csv(history_csv)
    history: Dict[str, np.ndarray] = {}
    for col in df.columns:
        history[col] = df[col].to_numpy()
    return history


def plot_training_curves(history: Dict[str, np.ndarray], out_path: Path) -> Path:
    """Plots train/val loss + val F1-macro + val mAP."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    epochs = np.arange(1, len(history.get('train_loss', [])) + 1)

    # Loss curves
    axes[0].plot(epochs, history.get('train_loss', []), label='train_loss')
    axes[0].plot(epochs, history.get('val_loss', []), label='val_loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Val F1 macro
    axes[1].plot(epochs, history.get('val_f1_macro', []), label='val_f1_macro')
    axes[1].set_title('Validation F1-Macro')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1')
    axes[1].grid(True, alpha=0.3)

    # Val mAP
    axes[2].plot(epochs, history.get('val_mAP', []), label='val_mAP')
    axes[2].set_title('Validation mAP')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('mAP')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('MIL Training Curves', fontsize=12)
    fig.tight_layout()

    out_file = out_path / 'mil_training_curves.png'
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    return out_file


def plot_multilabel_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> Path:
    """Plots 8 binary confusion matrices as a single grid image."""
    cms = multilabel_confusion_matrix(y_true, y_pred)  # [C, 2, 2]

    cols = 4
    rows = int(np.ceil(len(class_names) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 7))
    axes = np.array(axes).reshape(rows, cols)

    for i, name in enumerate(class_names):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        cm = cms[i]
        # cm layout: [[TN, FP],[FN, TP]]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'], fontsize=8)
        ax.set_yticklabels(['True 0', 'True 1'], fontsize=8)

        for (yy, xx), val in np.ndenumerate(cm):
            ax.text(xx, yy, str(int(val)), ha='center', va='center', fontsize=9)

    # Hide unused axes
    for j in range(len(class_names), rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis('off')

    fig.suptitle('Multi-label Confusion Matrices (per class)', fontsize=12)
    fig.tight_layout()

    out_file = out_path / 'mil_multilabel_confusion_matrices.png'
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    return out_file


def evaluate_checkpoint(
    csv_path: str,
    features_dir: str,
    checkpoint_path: str,
    batch_size: int,
    threshold: float,
    device: str,
) -> Dict[str, Any]:
    dataset = ANAFeatureDataset(csv_path=csv_path, features_dir=features_dir, split='test', normalize_features=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = ANAMILModel(input_dim=768, num_classes=8)
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels, _meta in loader:
            features = features.to(device)
            logits = model(features)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0).astype(np.int32)
    probs = _sigmoid(logits)
    y_pred = (probs >= threshold).astype(np.int32)

    # Headline metrics
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    class_aps = []
    supports = []
    for c in range(y_true.shape[1]):
        supports.append(int(y_true[:, c].sum()))
        try:
            ap = average_precision_score(y_true[:, c], probs[:, c])
        except Exception:
            ap = 0.0
        class_aps.append(float(ap))

    mAP = float(np.mean(class_aps))

    return {
        'threshold': float(threshold),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'f1_weighted': float(f1_weighted),
        'mAP': float(mAP),
        'class_aps': class_aps,
        'supports': supports,
        'classes': ANAFeatureDataset.ICAP_CLASSES,
        'y_true': y_true,
        'y_pred': y_pred,
    }


def write_presentation_report(metrics: Dict[str, Any], out_path: Path) -> Path:
    report_file = out_path / 'mil_eval_report.txt'

    lines = []
    lines.append('MIL Clinical Evaluation Summary')
    lines.append('================================')
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append('')

    lines.append('Headline (Test Split)')
    lines.append('---------------------')
    lines.append(f"Threshold: {metrics['threshold']:.2f}")
    lines.append(f"F1-Macro:  {metrics['f1_macro']:.4f}")
    lines.append(f"F1-Micro:  {metrics['f1_micro']:.4f}")
    lines.append(f"F1-Weighted: {metrics['f1_weighted']:.4f}")
    lines.append(f"mAP:       {metrics['mAP']:.4f}")
    lines.append('')

    lines.append('Per-Class AP and Support')
    lines.append('------------------------')
    lines.append(f"{'Class':<32} {'AP':>8} {'Support':>10}")
    lines.append('-' * 54)
    for name, ap, sup in zip(metrics['classes'], metrics['class_aps'], metrics['supports']):
        lines.append(f"{name:<32} {ap:>8.4f} {sup:>10d}")

    lines.append('')
    lines.append('Notes for presentation')
    lines.append('----------------------')
    lines.append('- Use F1-Macro as the headline metric (handles rare classes).')
    lines.append('- Use mAP to show ranking quality for co-occurring patterns.')
    lines.append('- Confusion matrices shown are per-class binary (TN/FP/FN/TP).')

    report_file.write_text('\n'.join(lines), encoding='utf-8')
    return report_file


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MIL checkpoint and generate plots + report.')
    parser.add_argument('--csv', default=DEFAULT_CSV)
    parser.add_argument('--features-dir', default=DEFAULT_FEATURES_DIR)
    parser.add_argument('--checkpoint', default='best_model.pt')
    parser.add_argument('--history-csv', default=str(Path('output') / 'mil_training_history.csv'))
    parser.add_argument('--outdir', default='output')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    out_path = _ensure_outdir(args.outdir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    metrics = evaluate_checkpoint(
        csv_path=args.csv,
        features_dir=args.features_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=device,
    )

    # Confusion matrices figure
    cm_png = plot_multilabel_confusion_matrices(
        y_true=metrics['y_true'],
        y_pred=metrics['y_pred'],
        class_names=metrics['classes'],
        out_path=out_path,
    )

    # Training curves (if history exists)
    history_csv = Path(args.history_csv)
    history = _load_training_history(history_csv)
    curves_png = None
    if history is not None:
        curves_png = plot_training_curves(history, out_path)

    # Clinical report
    report_txt = write_presentation_report(metrics, out_path)

    # Save JSON metrics (without y_true/y_pred arrays)
    metrics_small = {k: v for k, v in metrics.items() if k not in ('y_true', 'y_pred')}
    (out_path / 'mil_eval_metrics.json').write_text(json.dumps(metrics_small, indent=2), encoding='utf-8')

    print('\nSaved outputs:')
    print(f"- {cm_png}")
    if curves_png is not None:
        print(f"- {curves_png}")
    else:
        print(f"- (no training curves; missing {history_csv})")
    print(f"- {report_txt}")
    print(f"- {out_path / 'mil_eval_metrics.json'}")


if __name__ == '__main__':
    main()
