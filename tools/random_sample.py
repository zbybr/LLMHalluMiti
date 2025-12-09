import pandas as pd

df = pd.read_csv('gpt-5_hotpotqa_responses.csv', encoding="latin-1")

df_yes = df[df['is_hallucination'] == 'YES']
df_no_all = df[df['is_hallucination'] == 'NO']
df_no_sample = df_no_all.sample(n=25, random_state=42)

df_final = pd.concat([df_yes, df_no_sample], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

df_final.to_csv('filtered_sample.csv', index=False)

print("number of YES: ", len(df_yes))
print("number of NO: ", len(df_no_sample))
print("Final length: ", len(df_final))
