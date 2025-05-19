#!/usr/bin/env python3
"""
evaluating the trained LightGBM model on sparse test data
"""
import argparse, numpy as np
import lightgbm as lgb
from scipy import sparse

from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def main(model_path, test_X, test_y, plot_cm, cm_out):
    model = lgb.Booster(model_file=model_path)
    X = sparse.load_npz(test_X).astype(np.float32)
    y = np.load(test_y)
    preds = model.predict(X).argmax(axis=1)

    print("accuracy:", accuracy_score(y,preds))
    print("mcc:", matthews_corrcoef(y,preds))
    print("Mmacro-f1:", f1_score(y,preds,average="macro"))

    if plot_cm:
        cm = confusion_matrix(y,preds)
        plt.figure(figsize=(8,8))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(cm_out)
        print("Saved CM →", cm_out)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True)
    p.add_argument("--test_X",  required=True)
    p.add_argument("--test_y",  required=True)
    p.add_argument("--plot_cm", action="store_true")
    p.add_argument("--cm_out",  default="confusion_matrix.png")
    args = p.parse_args()
    main(args.model, args.test_X, args.test_y, args.plot_cm, args.cm_out)
