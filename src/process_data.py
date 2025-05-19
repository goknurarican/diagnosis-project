#!/usr/bin/env python3
"""
process_data.py

1) Explore CSVs
2) Clean & fill:
   - AGE, SEX
   - Binary / Categorical / Multi-choice evidences
   - Leave INITIAL_EVIDENCE & DIFFERENTIAL_DIAGNOSIS intact
3) Save cleaned Parquet
"""

import os, json, argparse, ast, gc
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

def load_json(path):
    return json.loads(Path(path).read_text())

def explore(df, name):
    print(f"\n=== {name} SET ===")
    print("shape:", df.shape)
    print("missing %:\n", (df.isna().mean()*100).round(2).head())
    print("top10 pathologies:\n", df["PATHOLOGY"].value_counts().head(10))
    # sample evidences
    sample = df["EVIDENCES"].dropna().head(100).apply(ast.literal_eval)
    flat = [e for sub in sample for e in sub]
    print("top evidences (sample):", pd.Series(flat).value_counts().head(5).to_dict())

def clean(df, evidences):
    # AGE
    df["age"] = df["AGE"].fillna(df["AGE"].median()).astype(np.uint8)
    # SEX
    df["sex"] = df["SEX"].map({"M":0,"F":1}).fillna(0).astype(np.uint8)

    mlb = MultiLabelBinarizer()
    add, drop = [], []

    for ev_key, meta in evidences.items():
        if ev_key not in df:
            continue
        typ = meta["data_type"].upper()  # 'B','C','M'
        if typ == "B":
            df[ev_key] = df[ev_key].fillna(0).astype(np.int8)
        elif typ == "C":
            df[ev_key] = df[ev_key].fillna("NA").astype("category")
        else:
            # multi-choice
            lists = df[ev_key].fillna("").apply(
                lambda x: x.split("|") if isinstance(x,str) else []
            )
            arr = mlb.fit_transform(lists)
            cols = [f"{ev_key}__{c}" for c in mlb.classes_]
            add.append(pd.DataFrame(arr, columns=cols, index=df.index))
            drop.append(ev_key)

    # one-hot categorical evidences
    cats = df.select_dtypes(["category"]).columns.tolist()
    if cats:
        df = pd.get_dummies(df, columns=cats, dummy_na=False)

    # concat multi-choice
    if add:
        df = pd.concat([df] + add, axis=1)
        df.drop(columns=drop, inplace=True)

    # keep INITIAL_EVIDENCE, DIFFERENTIAL_DIAGNOSIS for next stage
    return df

def main(inp_dir, out_dir):
    inp, out = Path(inp_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    evid = load_json(inp / "release_evidences.json")
    for split in ("train","validate","test"):
        csvf = inp/f"{split}.csv"
        if not csvf.exists(): continue

        df = pd.read_csv(csvf, engine="python", quotechar='"')
        explore(df, split)
        dfc = clean(df.copy(), evid)
        print(f"After clean {split}:", dfc.shape)

        dfc.to_parquet(out/f"{split}_clean.parquet",
                       compression="snappy", index=False)
        print(f"Saved → {split}_clean.parquet")

        del df, dfc; gc.collect()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    main(args.input_dir, args.output_dir)
