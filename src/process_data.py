"""
process_data.py
1)Exploring csv files
2) Cleaning & filling:
   - AGE, SEX
   - Binary / Categorical / Multi-choice evidences
   - Drop INITIAL_EVIDENCE & DIFFERENTIAL_DIAGNOSIS to prevent leakage
   - Normalize column names (lowercase)
3) Saving the cleaned parquet
"""

import os
import json
import argparse
import ast
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer


def load_json(path):
    return json.loads(Path(path).read_text())


def explore(df: pd.DataFrame, name: str):
    print(f"\n=== {name.upper()} SET ===")
    print(f"shape: {df.shape}")
    print("Missing %:\n", (df.isna().mean() * 100).round(2).head())
    if 'PATHOLOGY' in df.columns:
        print("Top 10 pathologies:\n", df["PATHOLOGY"].value_counts().head(10))
    if 'EVIDENCES' in df.columns:
        sample = df["EVIDENCES"].dropna().head(100).apply(ast.literal_eval)
        flat = [e for sub in sample for e in sub]
        print("Top evidences (sample):", pd.Series(flat).value_counts().head(5).to_dict())


def clean(df: pd.DataFrame, evidences: dict) -> pd.DataFrame:
    # AGE
    df["age"] = df["AGE"].fillna(df["AGE"].median()).astype(np.uint8)
    # SEX
    df["sex"] = df["SEX"].map({"M": 0, "F": 1}).fillna(0).astype(np.uint8)

    mlb = MultiLabelBinarizer()
    add, drop = [], []

    for ev_key, meta in evidences.items():
        if ev_key not in df:
            continue
        dtype = meta.get("data_type", "").upper()
        if dtype == 'B':
            df[ev_key] = df[ev_key].fillna(0).astype(np.int8)
        elif dtype == 'C':
            df[ev_key] = df[ev_key].fillna('NA').astype('category')
        else:
            lists = df[ev_key].fillna('').apply(
                lambda x: x.split('|') if isinstance(x, str) else []
            )
            arr = mlb.fit_transform(lists)
            cols = [f"{ev_key}__{c}" for c in mlb.classes_]
            add.append(pd.DataFrame(arr, columns=cols, index=df.index))
            drop.append(ev_key)

    #one-hot categorical
    cat_cols = df.select_dtypes(['category']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    if add:
        df = pd.concat([df] + add, axis=1)
        df.drop(columns=drop, inplace=True)

    #dropping leakage columns to prevent memorizing
    for col in ['INITIAL_EVIDENCE', 'DIFFERENTIAL_DIAGNOSIS']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    #droping the original AGE/SEX to avoid duplicates
    df.drop(columns=['AGE', 'SEX'], inplace=True, errors='ignore')

    #normalizing column names
    df.columns = [col.lower() for col in df.columns]


    return df


def main(input_dir: str, output_dir: str):
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    evidences = load_json(inp / 'release_evidences.json')

    for split in ['train', 'validate', 'test']:
        csv_path = inp / f"{split}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, engine='python', quotechar='"')
        explore(df, split)

        print(f"Cleaning {split} data...")
        df_clean = clean(df.copy(), evidences)
        print(f"After clean: {df_clean.shape}")

        out_path = out / f"{split}_clean.parquet"
        df_clean.to_parquet(out_path, compression='snappy', index=False)
        print(f"Saved → {out_path}\n")

        del df, df_clean
        gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Explore & clean DDXPlus data, drop leakage columns")
    parser.add_argument('--input-dir',  required=True, help='Path to raw CSV folder')
    parser.add_argument('--output-dir', required=True, help='Where to save cleaned Parquets')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
