import pandas as pd

df = pd.read_csv("gpt-oss-20b_truthfulqa1.3.csv", encoding="latin-1")
df.to_csv("gpt-oss-20b_truthfulqa1.3_utf8.csv", index=False, encoding="utf-8")