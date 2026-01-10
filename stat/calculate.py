import pandas as pd

# df = pd.read_csv("../gpt-4o_mutation_outputs_dataset20250926_hallucination.csv", encoding="latin-1")
df = pd.read_csv("../gpt-4o/outputs/gpt-4o_mutation_outputs_gpt-4o_dataset20251225_utf8_responses.csv", encoding="utf-8-sig")
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

# # recheck_hallucination_rate: origin_hallucination == YES
# recheck_hallu = df[(df['recheck_hallucination'] == 'YES')]
# recheck_hallu_ratio = len(recheck_hallu) / total_samples
# print(f"Recheck Hallu rate: {len(recheck_hallu)} / {total_samples} ({recheck_hallu_ratio:.2%})")

# successful repair by Majority Voting: origin_hallucination == YES and recheck_hallucination_mv == NO
successful_repair_mv = df[(df['is_hallucination'] == 'YES') & (df['recheck_hallucination_mv'] == 'NO')]
successful_repair_ratio_mv = len(successful_repair_mv) / len(hallucination)
print(f"Successful repair by Majority Voting: {len(successful_repair_mv)} / {len(hallucination)} ({successful_repair_ratio_mv:.2%})")

# unnecessary repair by Majority Voting: hallucination_check == YES and origin_hallucination == NO and recheck_hallucination == YES
unnecessary_repair_mv = df[(df['is_hallucination'] == 'NO') & (df['recheck_hallucination_mv'] == 'YES')]
unnecessary_repair_ratio_mv = len(unnecessary_repair_mv) / (total_samples - len(hallucination))
print(f"Unnecessary repair by Majority Voting: {len(unnecessary_repair_mv)} / {total_samples - len(hallucination)} ({unnecessary_repair_ratio_mv:.2%})")

# average token_cost and time_cost by Majority Voting
avg_token_cost_mv = df["token_cost_mv"].mean()
avg_time_cost_mv = df["time_cost_mv"].mean()
print(f"Average token cost by Majority Voting: {avg_token_cost_mv:.2f}")
print(f"Average time cost by Majority Voting: {avg_time_cost_mv:.4f} s")

# successful repair by Confidence Score: origin_hallucination == YES and recheck_hallucination_cs == NO
successful_repair_cs = df[(df['is_hallucination'] == 'YES') & (df['recheck_hallucination_cs'] == 'NO')]
successful_repair_ratio_cs = len(successful_repair_cs) / len(hallucination)
print(f"Successful repair by Confidence Score: {len(successful_repair_cs)} / {len(hallucination)} ({successful_repair_ratio_cs:.2%})")

# unnecessary repair by Confidence Score: hallucination_check == YES and origin_hallucination == NO and recheck_hallucination == YES
unnecessary_repair_cs = df[(df['is_hallucination'] == 'NO') & (df['recheck_hallucination_cs'] == 'YES')]
unnecessary_repair_ratio_cs = len(unnecessary_repair_cs) / (total_samples - len(hallucination))
print(f"Unnecessary repair by Confidence Score: {len(unnecessary_repair_cs)} / {total_samples - len(hallucination)} ({unnecessary_repair_ratio_cs:.2%})")

# average token_cost and time_cost by Confidence Score
avg_token_cost_cs = df["token_cost_cs"].mean()
avg_time_cost_cs = df["time_cost_cs"].mean()
print(f"Average token cost by Confidence Score: {avg_token_cost_cs:.2f}")
print(f"Average time cost by Confidence Score: {avg_time_cost_cs:.4f} s")

# successful repair by Ranking: origin_hallucination == YES and recheck_hallucination_ra == NO
successful_repair_ra = df[(df['is_hallucination'] == 'YES') & (df['recheck_hallucination_ra'] == 'NO')]
successful_repair_ratio_ra = len(successful_repair_ra) / len(hallucination)
print(f"Successful repair by Ranking: {len(successful_repair_ra)} / {len(hallucination)} ({successful_repair_ratio_ra:.2%})")

# unnecessary repair by Ranking: hallucination_check == YES and origin_hallucination == NO and recheck_hallucination == YES
unnecessary_repair_ra = df[(df['is_hallucination'] == 'NO') & (df['recheck_hallucination_ra'] == 'YES')]
unnecessary_repair_ratio_ra = len(unnecessary_repair_ra) / (total_samples - len(hallucination))
print(f"Unnecessary repair by Ranking: {len(unnecessary_repair_ra)} / {total_samples - len(hallucination)} ({unnecessary_repair_ratio_ra:.2%})")

# average token_cost and time_cost by Ranking
avg_token_cost_ra = df["token_cost_ra"].mean()
avg_time_cost_ra = df["time_cost_ra"].mean()
print(f"Average token cost by Ranking: {avg_token_cost_ra:.2f}")
print(f"Average time cost by Ranking: {avg_time_cost_ra:.4f} s")

# pass@6: pass@6 == YES
passat6 = df[(df['pass@6'] == 'YES') & (df['is_hallucination'] == 'YES')]
passat6_ratio = len(passat6) / len(hallucination)
print(f"pass@6: {len(passat6)} / {len(hallucination)} ({passat6_ratio:.2%})")
