#!/usr/bin/env python3
"""
train_mlp.py
– Load sparse X.npz & y.npy
– 5-fold stratified CV
– Train MLPClassifier on each fold
– Save model and loss curve for each fold
– Print cross-validation metrics
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from sklearn.neural_network import MLPClassifier


def train_mlp_fold(X, y, train_idx, val_idx, mlp_params, fold, out_dir):
    clf = MLPClassifier(**mlp_params)
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[val_idx])

    # Plot loss curve
    plt.figure()
    plt.plot(clf.loss_curve_, label="Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"MLP Loss Curve - Fold {fold}")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / f"mlp_loss_fold{fold}.png")
    plt.close()

    return clf, preds


def main(x_path, y_path, out_dir):
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)

    print("Loading data…")
    X = sparse.load_npz(x_path).astype(np.float32).toarray()
    y = np.load(y_path)
    print(f"X shape = {X.shape}, y shape = {y.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mlp_params = {
        'hidden_layer_sizes': (100,),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 1e-4,
        'learning_rate': 'adaptive',
        'max_iter': 200,
        'random_state': 42
    }

    results = {'acc': [], 'mcc': [], 'f1': []}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold} (MLP) ---")
        model, preds = train_mlp_fold(X, y, tr_idx, va_idx, mlp_params, fold, out)
        joblib.dump(model, out / f"mlp_fold{fold}.pkl")

        acc = accuracy_score(y[va_idx], preds)
        mcc = matthews_corrcoef(y[va_idx], preds)
        f1  = f1_score(y[va_idx], preds, average='macro')
        print(f"MLP Fold{fold} → ACC={acc:.4f}, MCC={mcc:.4f}, F1={f1:.4f}")
        results['acc'].append(acc)
        results['mcc'].append(mcc)
        results['f1'].append(f1)

    print("\n=== Cross-Validation Results ===")
    for metric, vals in results.items():
        mean, std = np.mean(vals), np.std(vals)
        print(f"MLP {metric.upper()}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-fold CV training for DDXPlus with MLP")
    parser.add_argument("--x",      required=True, help="Path to train_X.npz")
    parser.add_argument("--y",      required=True, help="Path to train_y.npy")
    parser.add_argument("--output", required=True, help="Directory to save fold models")
    args = parser.parse_args()
    main(args.x, args.y, args.output)
