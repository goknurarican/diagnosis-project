#!/usr/bin/env python3
"""
Evaluate trained MLP models (5-fold) on sparse test data
and optionally plot/save confusion matrices
"""

import argparse
import numpy as np
import joblib
from pathlib import Path
from scipy import sparse
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def evaluate(model_path, X, y, plot_cm=False, cm_out=None):
    model = joblib.load(model_path)
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    mcc = matthews_corrcoef(y, preds)
    f1  = f1_score(y, preds, average='macro')

    if plot_cm and cm_out:
        cm = confusion_matrix(y, preds)
        plt.figure(figsize=(8, 8))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix: {model_path.name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.savefig(cm_out)
        plt.close()
        print(f"Saved CM → {cm_out}")

    return acc, mcc, f1


def main(model_dir, test_X_path, test_y_path, plot_cm):
    model_paths = sorted(Path(model_dir).glob("mlp_fold*.pkl"))

    # Load test data once
    X = sparse.load_npz(test_X_path).astype(np.float32).toarray()
    y = np.load(test_y_path)

    accs, mccs, f1s = [], [], []

    for i, model_path in enumerate(model_paths):
        cm_out = None
        if plot_cm:
            cm_out = model_path.with_name(f"cm_{model_path.stem}.png")

        acc, mcc, f1 = evaluate(model_path, X, y, plot_cm, cm_out)
        print(f"{model_path.name}: ACC={acc:.4f}, MCC={mcc:.4f}, F1={f1:.4f}")
        accs.append(acc)
        mccs.append(mcc)
        f1s.append(f1)

    print("\n=== AVERAGE METRICS OVER FOLDS ===")
    print(f"ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"MCC: {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
    print(f"F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 5-fold MLP models on test set")
    parser.add_argument("--model-dir", required=True, help="Folder with mlp_foldX.pkl models")
    parser.add_argument("--test_X", required=True, help="Path to test_X.npz")
    parser.add_argument("--test_y", required=True, help="Path to test_y.npy")
    parser.add_argument("--plot_cm", action="store_true", help="Save confusion matrix images per fold")
    args = parser.parse_args()

    main(args.model_dir, args.test_X, args.test_y, args.plot_cm)
