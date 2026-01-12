import csv
import pandas as pd

df = pd.read_csv("input.csv", encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

mask = df["final_answer_cs"].astype(str).str.lower().str.contains("confidence score", na=False)
df_hit = df[mask].copy()

df_hit.to_csv("output.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
print(f"Saved {len(df_hit)} rows to output.csv")
