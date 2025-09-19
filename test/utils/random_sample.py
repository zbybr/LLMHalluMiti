import pandas as pd

input_path = "../hotpotqa_extended.csv"
df = pd.read_csv(input_path)

sample_df = df.sample(n=100, random_state=77)

output_path = "../hotpotqa_extended_sample.csv"
sample_df.to_csv(output_path, index=False)
