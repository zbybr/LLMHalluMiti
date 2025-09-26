import pandas as pd

df = pd.read_csv("gpt_outputs_dataset1.0_hallucination_checked_nsp.csv", encoding="latin-1")

total_samples = len(df)
print(f"Total samples: {total_samples}")

# 0. hallucination_rate: origin_hallucination == YES
hallucination = df[(df['origin_hallucination'] == 'YES')]
hallucination_ratio = len(hallucination) / total_samples
print(f"Successful detect: {len(hallucination)} ({hallucination_ratio:.2%})")

# 1. successful_detect: hallucination_check == YES 且 origin_hallucination == YES
successful_detect = df[(df['hallucination_check'] == 'YES') & (df['origin_hallucination'] == 'YES')]
successful_detect_ratio = len(successful_detect) / total_samples
print(f"Successful detect: {len(successful_detect)} ({successful_detect_ratio:.2%})")

# 2. unsuccessful_detect: hallucination_check == NO 且 origin_hallucination == YES
unsuccessful_detect = df[(df['hallucination_check'] == 'NO') & (df['origin_hallucination'] == 'YES')]
unsuccessful_detect_ratio = len(unsuccessful_detect) / total_samples
print(f"Unsuccessful detect: {len(unsuccessful_detect)} ({unsuccessful_detect_ratio:.2%})")

# 3. detect_ratio
detect_ratio = len(successful_detect) / len(hallucination)
print(f"Detect_ratio: {len(successful_detect)} / {len(hallucination)} ({detect_ratio:.2%})")

# 4. successful_repair: origin_hallucination == YES 且 recheck_hallucination == NO
successful_repair = df[(df['hallucination_check'] == 'YES') &
                       (df['origin_hallucination'] == 'YES') &
                       (df['recheck_hallucination'] == 'NO')]
successful_repair_ratio = len(successful_repair) / total_samples
print(f"Successful repair: {len(successful_repair)} ({successful_repair_ratio:.2%})")

# 5. unsuccessful_repair: hallucination_check == YES 且 origin_hallucination == YES 且 recheck_hallucination == YES
unsuccessful_repair = df[(df['hallucination_check'] == 'YES') &
                         (df['origin_hallucination'] == 'YES') &
                         (df['recheck_hallucination'] == 'YES')]
unsuccessful_repair_ratio = len(unsuccessful_repair) / total_samples
print(f"Unsuccessful repair: {len(unsuccessful_repair)} ({unsuccessful_repair_ratio:.2%})")

# 6. repair_ratio
repair_ratio = len(successful_repair) / len(successful_detect)
print(f"Detect_ratio: {len(successful_repair)} / {len(successful_detect)} ({repair_ratio:.2%})")

# 7. unnecessary_repair: hallucination_check == YES 且 origin_hallucination == NO 且 recheck_hallucination == YES
unnecessary_repair = df[(df['hallucination_check'] == 'YES') &
                        (df['origin_hallucination'] == 'NO') &
                        (df['recheck_hallucination'] == 'YES')]
unnecessary_repair_ratio = len(unnecessary_repair) / total_samples
print(f"Unnecessary repair: {len(unnecessary_repair)} ({unnecessary_repair_ratio:.2%})")


