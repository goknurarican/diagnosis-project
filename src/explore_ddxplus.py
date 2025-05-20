#!/usr/bin/env python3
"""
Exploratory Data Analysis for the DDXPlus medical dataset

This script:
- Summarizes dataset structure
- Plots distributions of age, sex, and pathologies
- Analyzes the most common evidence codes
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from pathlib import Path
import argparse

def load_data(path):
    df = pd.read_csv(path)
    return df

def plot_age_distribution(df, output_dir):
    plt.figure(figsize=(8,5))
    sns.histplot(df["AGE"].dropna(), bins=30, kde=True, color='skyblue')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "age_distribution.png")
    plt.close()

def plot_sex_distribution(df, output_dir):
    plt.figure(figsize=(5,4))
    sns.countplot(data=df, x="SEX", palette="Set2")
    plt.title("Sex Distribution")
    plt.xlabel("Sex")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "sex_distribution.png")
    plt.close()

def plot_pathology_distribution(df, output_dir):
    plt.figure(figsize=(10,6))
    top_pathologies = df["PATHOLOGY"].value_counts().head(20)
    sns.barplot(x=top_pathologies.values, y=top_pathologies.index, palette="viridis")
    plt.title("Top 20 Most Frequent Pathologies")
    plt.xlabel("Count")
    plt.ylabel("Pathology")
    plt.tight_layout()
    plt.savefig(output_dir / "top_pathologies.png")
    plt.close()

def plot_top_evidences(df, output_dir):
    evidence_series = df["EVIDENCES"].dropna().apply(ast.literal_eval)
    flat_evidences = [e for sublist in evidence_series for e in sublist]
    evidence_counts = Counter(flat_evidences)
    top_evidences = dict(evidence_counts.most_common(20))

    plt.figure(figsize=(10,6))
    sns.barplot(x=list(top_evidences.values()), y=list(top_evidences.keys()), palette="mako")
    plt.title("Top 20 Most Frequent Evidences")
    plt.xlabel("Count")
    plt.ylabel("Evidence Code")
    plt.tight_layout()
    plt.savefig(output_dir / "top_evidences.png")
    plt.close()

def generate_summary_table(df, output_dir):
    summary = {
        "Total Records": len(df),
        "Number of Features": len(df.columns),
        "Missing Values (%)": (df.isnull().mean() * 100).round(2).to_dict(),
        "Age Stats": df["AGE"].describe().round(2).to_dict(),
        "Sex Distribution": df["SEX"].value_counts().to_dict()
    }

    summary_path = output_dir / "dataset_summary.txt"
    with open(summary_path, "w") as f:
        for key, value in summary.items():
            f.write(f"{key}:\n{value}\n\n")
    print(f"Summary written to {summary_path}")

def main(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}")
    df = load_data(input_path)

    print("Generating plots and summary...")
    plot_age_distribution(df, output_path)
    plot_sex_distribution(df, output_path)
    plot_pathology_distribution(df, output_path)
    plot_top_evidences(df, output_path)
    generate_summary_table(df, output_path)

    print(f"All outputs saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and summarize the DDXPlus dataset")
    parser.add_argument("--input", required=True, help="Path to CSV file (train.csv, validate.csv, or test.csv)")
    parser.add_argument("--output", required=True, help="Directory to save visualizations and summary")
    args = parser.parse_args()
    main(args.input, args.output)
