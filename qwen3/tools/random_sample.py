import pandas as pd
import csv

df1 = pd.read_csv("gpt-4o_dataset20251225_utf8_responses_sampled.csv", encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
df2 = pd.read_csv("qwen3_32b_dataset20251225_utf8_responses.csv", encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

matched_df = df2[df2['Question'].isin(df1['Question'])]
final_df = matched_df[['Question', 'Answer', 'base_response']]

final_df.to_csv("sampled_output.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
print("Sampling finished. Saved to sampled_output.csv")
