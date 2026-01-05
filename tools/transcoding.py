import pandas as pd

df = pd.read_csv("dataset20251225_utf8.csv", encoding="latin-1")
df.to_csv("dataset20251225_utf8.csv", index=False, encoding="utf-8-sig")