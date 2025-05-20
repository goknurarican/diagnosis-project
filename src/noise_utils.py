
import ast, random
import numpy as np
import pandas as pd
def random_drop_evidence(ev_str, p=0.1):
    if pd.isna(ev_str): return ev_str
    lst = ast.literal_eval(ev_str)
    kept = [c for c in lst if random.random() > p]
    return str(kept)
