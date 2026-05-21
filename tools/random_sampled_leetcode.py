import pandas as pd

df = pd.read_csv('leetcode_train.csv')
sample_size = min(200, len(df))
df_sampled = df.sample(n=sample_size, random_state=42)
df_sampled.to_csv('leetcode_sampled.csv', index=False, encoding='utf-8-sig')
