#!/usr/bin/env python3
"""
train.py
– Load sparse X.npz & y.npy
– 5-fold stratified CV
– Train LightGBM on each fold with stronger regularization
– Save each fold model & report mean metrics
"""
import argparse
import numpy as np
import lightgbm as lgb
from pathlib import Path
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score


def train_fold(X, y, train_idx, val_idx, params, num_rounds, callbacks):
    dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
    dval   = lgb.Dataset(X[val_idx],   label=y[val_idx], reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train","val"],
        num_boost_round=num_rounds,
        callbacks=callbacks
    )
    preds = booster.predict(X[val_idx]).argmax(axis=1)
    return booster, preds


def main(x_path, y_path, out_dir, use_gpu=False):
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)

    print("Loading data…")
    X = sparse.load_npz(x_path).astype(np.float32)
    y = np.load(y_path)
    print(f"X shape = {X.shape}, y shape = {y.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # LightGBM parameters with regularization
    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y)),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_bin": 255,
        # regularization
        "feature_fraction": 0.3,
        "bagging_fraction": 0.3,
        "bagging_freq": 1,
        "max_depth": 6,
        "min_child_samples": 100,
        "verbosity": -1,
        "seed": 42,
        "class_weight": "balanced",
        "device_type": "gpu" if use_gpu else "cpu"
    }

    callbacks = [
        lgb.early_stopping(20),
        lgb.log_evaluation(10)
    ]

    results = {"acc": [], "mcc": [], "f1": []}

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold} ---")
        model, preds = train_fold(X, y, tr, va, params, num_rounds=80, callbacks=callbacks)
        model.save_model(out / f"fold{fold}.txt")
        acc = accuracy_score(y[va], preds)
        mcc = matthews_corrcoef(y[va], preds)
        f1  = f1_score(y[va], preds, average="macro")
        print(f"Fold{fold} → ACC={acc:.4f}, MCC={mcc:.4f}, F1={f1:.4f}")
        results["acc"].append(acc)
        results["mcc"].append(mcc)
        results["f1"].append(f1)

    print("\n=== Cross-Validation Results ===")
    for metric, vals in results.items():
        print(f"{metric.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-fold CV training for DDXPlus")
    parser.add_argument("--x",      required=True, help="Path to train_X.npz")
    parser.add_argument("--y",      required=True, help="Path to train_y.npy")
    parser.add_argument("--output", required=True, help="Directory to save fold models")
    parser.add_argument("--gpu",    action="store_true", help="Use GPU if available")
    args = parser.parse_args()
    main(args.x, args.y, args.output, args.gpu)
