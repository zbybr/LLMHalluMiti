import pandas as pd

df = pd.read_csv('leetcode_train.csv')
df_hard = df[df['difficulty'].str.lower() == 'hard']
df_hard.to_csv('leetcode_hard.csv', index=False, encoding='utf-8-sig')

print(len(df_hard))