import pandas as pd
import ast
from collections import Counter

# 1) Parquet’i yükle
df = pd.read_parquet("data/processed/train_clean.parquet")

# 2) Evidences kolonunu explode edip tek bir listeye indir
all_codes = []
for ev_str in df['evidences'].dropna():
    all_codes.extend(ast.literal_eval(ev_str))

# 3) Frekansları say
counts = Counter(all_codes)

top100 = [code for code, _ in counts.most_common(100)]
print("Top 100 codes:", top100)
