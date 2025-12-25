import pandas as pd

df = pd.read_csv("merged_answers.csv", encoding="iso-8859-1")
df_selected = df.sample(n=1067, random_state=42)
df_selected.to_csv("hotpotqa_sampled.csv", index=False, encoding="iso-8859-1")