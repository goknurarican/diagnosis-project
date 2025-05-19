#!/usr/bin/env python3
"""
feature_build.py

– Reads *_clean.parquet (with INITIAL_EVIDENCE &
  DIFFERENTIAL_DIAGNOSIS columns)
– Builds ev2id.json on train (reused on other splits)
– Bins age, encodes sex
– Multi-hot evidences + initial evidence one-hot
– Extracts first 3 differential probs & label-IDs
– Saves X.npz, y.npy
"""

import os, json, ast, argparse
import numpy as np, pandas as pd
from pathlib import Path
from scipy import sparse

def make_ev2id(df):
    codes = set()
    for s in df["evidences"].dropna():
        codes |= set(ast.literal_eval(s))
    return {c:i for i,c in enumerate(sorted(codes))}

def load_ev2id(path):
    return json.loads(Path(path).read_text())

def bin_age(arr):
    bins = [0,15,41,66, np.inf]
    idx = np.digitize(arr, bins)-1
    mat = np.zeros((len(arr),4),dtype=np.int8)
    mat[np.arange(len(arr)), idx] = 1
    return mat

def build_features(df, ev2id, cond2id):
    n, m = len(df), len(ev2id)
    # 1) age bins & sex
    age_mat = bin_age(df["age"].to_numpy())
    sex_mat = df["sex"].to_numpy().reshape(-1,1).astype(np.int8)

    # 2) multi-hot evidences
    rows,cols = [],[]
    for i,s in enumerate(df["evidences"]):
        if pd.isna(s): continue
        for code in ast.literal_eval(s):
            j = ev2id.get(code)
            if j is not None:
                rows.append(i); cols.append(j)
    data = np.ones(len(rows),dtype=np.int8)
    ev_mat = sparse.csr_matrix((data,(rows,cols)),shape=(n,m),dtype=np.int8)

    # 3) initial evidence one-hot
    init_rows, init_cols = [], []
    for i,code in enumerate(df["initial_evidence"]):
        j = ev2id.get(code)
        if j is not None:
            init_rows.append(i); init_cols.append(j)
    init_data = np.ones(len(init_rows),dtype=np.int8)
    init_mat = sparse.csr_matrix(
        (init_data,(init_rows,init_cols)), shape=(n,m), dtype=np.int8
    )

    # 4) differential diagnosis (first 3)
    diff_probs = np.zeros((n,3),dtype=np.float32)
    diff_ids   = np.zeros((n,3),dtype=np.int16)
    for i,s in enumerate(df["differential_diagnosis"]):
        if pd.isna(s): continue
        arr = ast.literal_eval(s)  # list of [ [cond,prob],... ]
        for k,(cond,prob) in enumerate(arr[:3]):
            diff_probs[i,k] = prob
            diff_ids[i,k]   = cond2id.get(cond, -1)

    # assemble dense + sparse
    dense = np.hstack([age_mat, sex_mat, diff_probs, diff_ids])
    X = sparse.hstack([
        sparse.csr_matrix(dense),
        ev_mat, init_mat
    ], format="csr")
    return X

def main(input_path, output_dir, ev2id_path=None, cond2id_path=None):
    inp = Path(input_path)
    out = Path(output_dir); out.mkdir(exist_ok=True, parents=True)
    df  = pd.read_parquet(inp)

    # load or build ev2id
    if ev2id_path:
        ev2id = load_ev2id(ev2id_path)
    else:
        ev2id = make_ev2id(df)
        (out/"ev2id.json").write_text(json.dumps(ev2id,indent=2))
        print(f"Built ev2id ({len(ev2id)})")

    # load or build cond2id (pathology codes)
    if cond2id_path:
        cond2id = json.loads(Path(cond2id_path).read_text())
    else:
        cats = df["pathology"].astype("category").cat.categories
        cond2id = {c:i for i,c in enumerate(cats)}
        (out/"cond2id.json").write_text(json.dumps(cond2id,indent=2))
        print(f"Built cond2id ({len(cond2id)})")

    X = build_features(df, ev2id, cond2id)
    basename = inp.stem.replace("_clean","")
    npz = out/f"{basename}_X.npz"
    yfn = out/f"{basename}_y.npy"
    sparse.save_npz(npz, X)
    np.save(yfn, df["pathology"].astype("category").cat.codes.to_numpy())
    print(f"Saved → {npz} ({X.shape}), {yfn}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ev2id",      default=None)
    p.add_argument("--cond2id",    default=None)
    args = p.parse_args()
    main(args.input, args.output_dir, args.ev2id, args.cond2id)
