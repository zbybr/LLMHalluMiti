import pandas as pd

# df = pd.read_csv("../gpt-4o_mutation_outputs_dataset20250926_hallucination.csv", encoding="latin-1")
df = pd.read_csv("../outputs/gpt-oss_20b_outputs_gpt-oss-20b_truthfulqa1.3.csv", encoding="latin-1")
total_samples = len(df)
print(f"Total samples: {total_samples}")


def normalize(value):
    if not isinstance(value, str):
        return value
    v = value.upper().strip()
    if v.startswith("YES"):
        return "YES"
    elif v.startswith("NO"):
        return "NO"
    else:
        return v


# df['hallucination_check'] = df['hallucination_check'].apply(normalize)

# hallucination_rate: origin_hallucination == YES
hallucination = df[(df['is_hallucination'] == 'YES')]
hallucination_ratio = len(hallucination) / total_samples
print(f"Hallucination rate: {len(hallucination)} / {total_samples} ({hallucination_ratio:.2%})")

# re-hallucination_rate: origin_hallucination == YES
recheck_hallu = df[(df['recheck_hallucination'] == 'YES')]
recheck_hallu_ratio = len(recheck_hallu) / total_samples
print(f"Recheck Hallu rate: {len(recheck_hallu)} / {total_samples} ({recheck_hallu_ratio:.2%})")

# # successful_detect: ['base_response'] != ['proco_answer']
# successful_detect = df[(df['origin_hallucination'] == 'YES') & (df['base_response'] != df['proco_answer'])]
# successful_detect_ratio = len(successful_detect) / len(hallucination)
# print(f"Successful detect: {len(successful_detect)} / {len(hallucination)}  ({successful_detect_ratio:.2%})")

# # unsuccessful_detect: hallucination_check == NO and origin_hallucination == YES
# unsuccessful_detect = df[(df['hallucination_check'] == 'NO') & (df['origin_hallucination'] == 'YES')]
# unsuccessful_detect_ratio = len(unsuccessful_detect) / total_samples
# print(f"Unsuccessful detect: {len(unsuccessful_detect)} ({unsuccessful_detect_ratio:.2%})")

# # detect_ratio
# detect_ratio = len(successful_detect) / len(hallucination)
# print(f"Detect_ratio: {len(successful_detect)} / {len(hallucination)} ({detect_ratio:.2%})")

# successful_repair: origin_hallucination == YES and recheck_hallucination == NO
successful_repair = df[(df['is_hallucination'] == 'YES') & (df['recheck_hallucination'] == 'NO')]
successful_repair_ratio = len(successful_repair) / len(hallucination)
print(f"Successful repair: {len(successful_repair)} / {len(hallucination)} ({successful_repair_ratio:.2%})")

# unsuccessful_repair: origin_hallucination == YES and base_response != proco_answer and recheck_hallucination == YES
unsuccessful_repair = df[
    (df['is_hallucination'] == 'YES') &
    (df['recheck_hallucination'] == 'YES')
]
unsuccessful_repair_ratio = len(unsuccessful_repair) / len(hallucination)
print(f"Unsuccessful repair: {len(unsuccessful_repair)} / {len(hallucination)} ({unsuccessful_repair_ratio:.2%})")

# # unable_repair: origin_hallucination == YES and base_response == proco_answer
# unable_repair = df[
#     (df['is_hallucination'] == 'YES') &
#     (df['base_response'] == df['proco_answer'])
# ]
# unable_repair_ratio = len(unable_repair) / len(hallucination)
# print(f"Unable repair: {len(unable_repair)} / {len(hallucination)} ({unable_repair_ratio:.2%})")


# unnecessary_repair: hallucination_check == YES and origin_hallucination == NO and recheck_hallucination == YES
unnecessary_repair = df[(df['is_hallucination'] == 'NO') & (df['recheck_hallucination'] == 'YES')]
unnecessary_repair_ratio = len(unnecessary_repair) / (total_samples - len(hallucination))
print(f"Unnecessary repair: {len(unnecessary_repair)} / {total_samples - len(hallucination)} ({unnecessary_repair_ratio:.2%})")

# average token_cost and time_cost
avg_token_cost = df["token_cost"].mean()
avg_time_cost = df["time_cost"].mean()
print(f"Average token cost: {avg_token_cost:.2f}")
print(f"Average time cost: {avg_time_cost:.4f} s")
