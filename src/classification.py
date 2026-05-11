import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.data_loader import parse_filename, DATA_DIR

def build_dataset(data_dir=DATA_DIR, max_samples=None):
    """
    Loads all global CLS tokens and corresponding labels to build a tabular dataset.
    """
    files = list(data_dir.glob("*.npy"))
    if max_samples:
        files = files[:max_samples]
        
    X = []
    y = []
    
    print(f"Building dataset from {len(files)} files...")
    for f in tqdm(files, desc="Extracting CLS tokens"):
        meta = parse_filename(f.name)
        pattern = meta['pattern']
        
        try:
            feats = np.load(f)
            cls_token = feats[0]  # The global representation
            X.append(cls_token)
            y.append(pattern)
        except Exception:
            pass
            
    return np.array(X), np.array(y)

def train_and_evaluate():
    """
    Trains a lightweight classifier on the CLS tokens and evaluates its accuracy.
    """
    X, y = build_dataset()
    
    if len(X) == 0:
        print("No data loaded. Aborting classification.")
        return
        
    # Get class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"  {cls}: {count}")
    
    # Filter out rare classes if we are doing k-fold or stratification where classes need >=2 samples.
    print(f"\nFiltering out classes with fewer than 2 samples to allow stratification...")
    valid_classes = unique_classes[counts >= 2]
    
    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    
    print(f"Data after filtering: {X.shape}")

    print("\nSplitting dataset into 80% train, 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    print("\nTraining HistGradientBoostingClassifier (LightGBM equivalent)...")
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Direct Classification Accuracy (CLS Token only): {acc*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    train_and_evaluate()
