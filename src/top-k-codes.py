import pandas as pd
import ast
from collections import Counter

df = pd.read_parquet("data/processed/train_clean.parquet")
all_codes = []
for ev_str in df['evidences'].dropna():
    all_codes.extend(ast.literal_eval(ev_str))
counts = Counter(all_codes)

top100 = [code for code, _ in counts.most_common(100)]
print("Top 100 codes:", top100)
