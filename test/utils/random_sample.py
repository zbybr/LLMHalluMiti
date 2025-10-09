import pandas as pd
from pathlib import Path

datasets = {
    "truthfulqa": {"path": "../truthfulqa1.3.csv", "sample_size": 85},
    "hotpotqa": {"path": "../hotpotqa.csv", "sample_size": 76},
    "triviaqa": {"path": "../triviaqa_semicolon.csv", "sample_size": 63},
}

RANDOM_SEED = 77

for name, info in datasets.items():
    df = pd.read_csv(info["path"], encoding="latin-1")

    sample_df = df.sample(n=info["sample_size"], random_state=RANDOM_SEED)

    output_path = Path(f"{name}_sampled.csv")
    sample_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"{name} to {output_path}")
