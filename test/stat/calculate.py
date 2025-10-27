import pandas as pd

df = pd.read_csv("../gpt-4o_outputs_dataset20250926_hallucination.csv", encoding="latin-1")

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


df['is_hallucination'] = df['is_hallucination'].apply(normalize)

# hallucination_rate: origin_hallucination == YES
hallucination = df[(df['is_hallucination'] == 'YES')]
hallucination_ratio = len(hallucination) / total_samples
print(f"Hallucination rate: {len(hallucination)} ({hallucination_ratio:.2%})")

# re-hallucination_rate: origin_hallucination == YES
recheck_hallu = df[(df['recheck_hallucination'] == 'YES')]
recheck_hallu_ratio = len(recheck_hallu) / total_samples
print(f"Recheck Hallu rate: {len(recheck_hallu)} ({recheck_hallu_ratio:.2%})")


# successful_repair: hallucination_check == YES and is_hallucination == YES and recheck_hallucination == NO
successful_repair = df[(df['hallucination_check'] == 'YES') &
                       (df['is_hallucination'] == 'YES') &
                       (df['recheck_hallucination'] == 'NO')]
successful_repair_ratio = len(successful_repair) / len(hallucination)
print(f"Successful repair: {len(successful_repair)} / {len(hallucination)} ({successful_repair_ratio:.2%})")

# unsuccessful_repair: hallucination_check == YES and is_hallucination == YES and recheck_hallucination == YES
unsuccessful_repair = df[(df['hallucination_check'] == 'YES') &
                         (df['is_hallucination'] == 'YES') &
                         (df['recheck_hallucination'] == 'YES')]
unsuccessful_repair_ratio = len(unsuccessful_repair) / len(hallucination)
print(f"Unsuccessful repair: {len(unsuccessful_repair)} / {len(hallucination)} ({unsuccessful_repair_ratio:.2%})")

# unable_repair: hallucination_check == NO and is_hallucination == YES
unable_repair = df[(df['is_hallucination'] == 'YES') & (df['hallucination_check'] == 'NO')]
unable_repair_ratio = len(unable_repair) / len(hallucination)
print(f"Unable repair: {len(unable_repair)} / {len(hallucination)} ({unable_repair_ratio:.2%})")

# repair_ratio
repair_ratio = len(successful_repair) / len(hallucination)
print(f"Repair_ratio: {len(successful_repair)} / {len(hallucination)} ({repair_ratio:.2%})")

# unnecessary_repair: hallucination_check == YES and origin_hallucination == NO and recheck_hallucination == YES
unnecessary_repair = df[(df['hallucination_check'] == 'YES') &
                        (df['is_hallucination'] == 'NO') &
                        (df['recheck_hallucination'] == 'YES')]
unnecessary_repair_ratio = len(unnecessary_repair) / total_samples
print(f"Unnecessary repair: {len(unnecessary_repair)} ({unnecessary_repair_ratio:.2%})")

# average token_cost and time_cost
avg_token_cost = df["token_cost"].mean()
avg_time_cost = df["time_cost"].mean()
print(f"Average token cost: {avg_token_cost:.2f}")
print(f"Average time cost: {avg_time_cost:.4f} s")
