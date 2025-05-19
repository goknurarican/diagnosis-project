#!/usr/bin/env python3
"""
train.py
– loading train_X.npz ve train_y.npy
– 5-fold stratified cv
– training LightGBM on each fold
– saving each fold model & report mean metrics
"""

import argparse
import numpy as np
import lightgbm as lgb
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

def train_fold(X, y, train_idx, val_idx, params, num_rounds, callbacks):
    dtrain=lgb.Dataset(X[train_idx], label=y[train_idx])
    dval= lgb.Dataset(X[val_idx],   label=y[val_idx], reference=dtrain)
    booster = lgb.train(
        params, dtrain, valid_sets=[dtrain, dval],
        valid_names=["train","val"],

        num_boost_round=num_rounds,
        callbacks=callbacks
    )
    preds= booster.predict(X[val_idx]).argmax(axis=1)
    return booster, preds

def main(x_path, y_path, out_dir, use_gpu=False):
    out = Path(out_dir); out.mkdir(exist_ok=True, parents=True)
    X =sparse.load_npz(x_path).astype(np.float32)
    y =np.load(y_path)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    params = {
      "objective":"multiclass", "num_class":len(np.unique(y)),
      "metric":"multi_logloss",
      "boosting_type":"gbdt","learning_rate":0.05,
      "num_leaves":64,"max_bin":255,"verbosity":-1,
      "seed":42,"class_weight":"balanced",
      "device_type":"gpu" if use_gpu else "cpu"
    }
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
    results = {"acc":[],"mcc":[],"f1":[]}

    for fold,(tr,va) in enumerate(skf.split(X,y)):
        print(f"\n--- fold {fold} ---")
        model, preds = train_fold(X,y,tr,va,params,500,callbacks)
        # save
        model.save_model(out/f"fold{fold}.txt")
        # metrics
        acc = accuracy_score(y[va], preds)
        mcc = matthews_corrcoef(y[va], preds)
        f1  = f1_score(y[va], preds, average="macro")
        print(f"fold{fold} → acc={acc:.4f}, MCC={mcc:.4f}, F1={f1:.4f}")
        results["acc"].append(acc)
        results["mcc"].append(mcc)
        results["f1"].append(f1)

    print("\n cv results:")
    for k,v in results.items():
        print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

if __name__=="__main__":
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--x",        required=True)
    p.add_argument("--y",        required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--gpu",      action="store_true")
    args = p.parse_args()
    main(args.x, args.y, args.output, args.gpu)
