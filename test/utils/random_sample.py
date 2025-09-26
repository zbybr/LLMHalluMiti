import pandas as pd
from pathlib import Path

# 数据集路径和对应抽样数
datasets = {
    "dataset1": {"path": "../truthfulqa1.3.csv", "sample_size": 85},
    "dataset2": {"path": "../hotpotqa.csv", "sample_size": 76},
    "dataset3": {"path": "../triviaqa.csv", "sample_size": 63},
}

RANDOM_SEED = 77

for name, info in datasets.items():
    df = pd.read_csv(info["path"], encoding="latin-1")

    sample_df = df.sample(n=info["sample_size"], random_state=RANDOM_SEED)

    output_dir = Path(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{name}_sampled.csv"
    sample_df.to_csv(output_path, index=False, encoding="latin-1")

    print(f"{name} to {output_path}")
