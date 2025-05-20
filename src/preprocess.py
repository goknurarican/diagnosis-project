import pandas as pd
import numpy as np
import json
import argparse
from sklearn.preprocessing import MultiLabelBinarizer

def load_evidences(path):
    with open(path) as f:
        return json.load(f)

def clean_encode(df, evidences):
    #age imputation
    df['age'] = df['age'].fillna(df['age'].median())

    #sex to binary 0/1
    df['sex'] = df['sex'].map({'M':0, 'F':1}).astype('int8')

    #encoding
    mlb = MultiLabelBinarizer()
    to_drop = []

    for ev_key, ev_meta in evidences.items():
        col = ev_key
        if col not in df.columns:
            continue

        dtype = ev_meta['type']
        if dtype == 'binary':
            df[col] = df[col].fillna(0).astype('int8')

        elif dtype == 'categorical':
            #label encoding first, then one-hot
            df[col] = df[col].fillna('NA').astype('category')

        elif dtype == 'multi_choice':
            #split, multilabel binarize

            lists = df[col].fillna("").str.split('|')
            bin_df = pd.DataFrame(
                mlb.fit_transform(lists),
                columns=[f"{col}__{c}" for c in mlb.classes_],
                index=df.index
            )
            df = pd.concat([df, bin_df], axis=1)
            to_drop.append(col)

    #categorical columns to one-hot
    cat_cols = [c for c in df.select_dtypes(['category']).columns]
    df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dummy_na=False)

    #droping
    df.drop(columns=to_drop, inplace=True)
    return df

def main(input_csv, evidences_json, output_path):
    print("Loading data...")
    df = pd.read_csv(input_csv)
    print(f"original shape = {df.shape}")

    print("Loading evidences...")
    ev = load_evidences(evidences_json)

    print("Cleaning & encoding...")
    df_clean = clean_encode(df, ev)
    print(f"After encode shape = {df_clean.shape}")

    print(f"Saving to {output_path} …")
    df_clean.to_parquet(output_path, index=False)
    print("finish")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",    required=True, help="train/validate/test CSV")
    p.add_argument("--evidences",required=True, help="release_evidences.json yolu")
    p.add_argument("--output",   required=True, help="Çıktı Parquet dosyası")
    args = p.parse_args()
    main(args.input, args.evidences, args.output)
