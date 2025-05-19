#!/usr/bin/env python3
"""
feature_build.py

1) Load cleaned parquet (without leakage columns)
2) Mask top-k evidences and apply random noise
3) Build or load ev2id.json
4) Bin age into 4 one-hot bins, encode sex
5) Multi-hot encode evidences only
6) Save X as sparse .npz and y as .npy
"""
import json
import ast
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

# List of top codes to mask (based on train set frequency)
TOP_K_CODES =  ['E_204_@_V_10', 'E_53', 'E_57_@_V_123', 'E_66', 'E_201', 'E_54_@_V_161', 'E_79', 'E_54_@_V_192', 'E_91', 'E_181', 'E_131_@_V_10', 'E_129', 'E_54_@_V_179', 'E_54_@_V_181', 'E_55_@_V_89', 'E_54_@_V_183', 'E_135_@_V_12', 'E_124', 'E_54_@_V_154', 'E_55_@_V_29', 'E_50', 'E_78', 'E_55_@_V_101', 'E_55_@_V_159', 'E_55_@_V_55', 'E_55_@_V_197', 'E_55_@_V_62', 'E_58_@_6', 'E_56_@_5', 'E_54_@_V_198', 'E_59_@_5', 'E_55_@_V_166', 'E_70', 'E_148', 'E_55_@_V_160', 'E_55_@_V_56', 'E_59_@_4', 'E_59_@_3', 'E_58_@_5', 'E_56_@_4', 'E_58_@_7', 'E_45', 'E_56_@_6', 'E_48', 'E_151', 'E_54_@_V_180', 'E_49', 'E_77', 'E_56_@_7', 'E_56_@_3', 'E_55_@_V_148', 'E_55_@_V_167', 'E_130_@_V_156', 'E_97', 'E_58_@_4', 'E_59_@_2', 'E_136_@_0', 'E_144', 'E_41', 'E_82', 'E_89', 'E_69', 'E_59_@_0', 'E_56_@_8', 'E_54_@_V_182', 'E_58_@_2', 'E_58_@_3', 'E_58_@_8', 'E_59_@_1', 'E_57_@_V_39', 'E_104', 'E_155', 'E_226', 'E_55_@_V_20', 'E_55_@_V_21', 'E_55_@_V_137', 'E_105', 'E_55_@_V_109', 'E_55_@_V_108', 'E_55_@_V_25', 'E_56_@_2', 'E_214', 'E_123', 'E_88', 'E_222', 'E_218', 'E_55_@_V_170', 'E_55_@_V_33', 'E_208', 'E_227', 'E_55_@_V_171', 'E_59_@_6', 'E_220', 'E_51', 'E_116', 'E_76', 'E_55_@_V_124', 'E_132_@_0', 'E_135_@_V_10', 'E_209']


# Noise probability
NOISE_P = 0.4  # drop 40% of evidences randomly


def make_ev2id(df):
    codes = set()
    for s in df['evidences'].dropna():
        codes |= set(ast.literal_eval(s))
    # mask top codes before building id mapping
    codes -= set(TOP_K_CODES)
    return {c: i for i, c in enumerate(sorted(codes))}


def load_ev2id(path):
    return json.loads(Path(path).read_text())


def load_cond2id(path):
    return json.loads(Path(path).read_text())


def bin_age(arr):
    bins = [0, 15, 41, 66, np.inf]
    idx = np.digitize(arr, bins) - 1
    mat = np.zeros((len(arr), 4), dtype=np.int8)
    mat[np.arange(len(arr)), idx] = 1
    return mat


def random_mask_and_noise(s):
    if pd.isna(s):
        return s
    lst = ast.literal_eval(s)
    # mask top codes
    filtered = [c for c in lst if c not in TOP_K_CODES]
    # random noise drop
    noisy = [c for c in filtered if random.random() > NOISE_P]
    return str(noisy)


def build_features(df, ev2id, cond2id=None):
    # apply mask and noise to evidences
    df['evidences'] = df['evidences'].apply(random_mask_and_noise)

    n = len(df)
    m = len(ev2id)
    # 1) age bins + sex
    age_mat = bin_age(df['age'].to_numpy())
    sex_mat = df['sex'].to_numpy().reshape(-1, 1).astype(np.int8)

    # 2) multi-hot evidences
    rows, cols = [], []
    for i, s in enumerate(df['evidences']):
        if pd.isna(s):
            continue
        for code in ast.literal_eval(s):
            j = ev2id.get(code)
            if j is not None:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.int8)
    ev_mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, m), dtype=np.int8)

    # assemble dense + sparse
    dense = np.hstack([age_mat, sex_mat])
    X = sparse.hstack([sparse.csr_matrix(dense), ev_mat], format='csr')
    return X


def main(input_path, output_dir, ev2id_path=None, cond2id_path=None):
    inp = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(inp)

    # ev2id
    if ev2id_path:
        ev2id = load_ev2id(ev2id_path)
    else:
        ev2id = make_ev2id(df)
        (out / 'ev2id.json').write_text(json.dumps(ev2id, indent=2))
        print(f'Built ev2id ({len(ev2id)})')

    # cond2id
    if cond2id_path:
        cond2id = load_cond2id(cond2id_path)
    else:
        cats = df['pathology'].astype('category').cat.categories
        cond2id = {c: i for i, c in enumerate(cats)}
        (out / 'cond2id.json').write_text(json.dumps(cond2id, indent=2))
        print(f'Built cond2id ({len(cond2id)})')

    # build features
    X = build_features(df, ev2id, cond2id)
    basename = inp.stem.replace('_clean', '')
    X_path = out / f'{basename}_X.npz'
    y_path = out / f'{basename}_y.npy'
    sparse.save_npz(X_path, X)
    np.save(y_path, df['pathology'].astype('category').cat.codes.to_numpy())
    print(f'Saved → {X_path} ({X.shape}), {y_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build features without leakage columns')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ev2id', default=None)
    parser.add_argument('--cond2id', default=None)
    args = parser.parse_args()
    main(args.input, args.output_dir, args.ev2id, args.cond2id)
