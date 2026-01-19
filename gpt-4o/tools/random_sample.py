import pandas as pd
import csv

df = pd.read_csv("gpt-4o_dataset20251225_utf8_responses.csv", encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

part1 = df.iloc[1:818]
part2 = df.iloc[818:1217]
part3 = df.iloc[1217:2827]

sample1 = part1.sample(n=80, random_state=42)
sample2 = part2.sample(n=40, random_state=42)
sample3 = part3.sample(n=60, random_state=42)

final_df = pd.concat([sample1, sample2, sample3], ignore_index=True)
final_df.to_csv("sampled_output.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
print("Sampling finished. Saved to sampled_output.csv")
