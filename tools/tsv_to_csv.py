import pandas as pd

df = pd.read_csv('Poly-FEVER.tsv', sep='\t', encoding='utf-8')
df.to_csv('Poly-FEVER.csv', index=False, encoding='utf-8')
