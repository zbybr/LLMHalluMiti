import argparse
import csv
from tqdm import tqdm
from pathlib import Path
import prompt
import pandas as pd


def run_pipeline(input_path, output_path):
    df = pd.read_csv(input_path, encoding="latin-1", quoting=csv.QUOTE_ALL)

    print(f"Loaded dataset '{input_path}' with {len(df)} questions.")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        question = row["Question"]
        answer = row["Answer"]
        adv_final, adv_tokens, adv_time = prompt.run_advanced_proco_pipeline(question, args.model_key)

        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Final Answer: {adv_final} (extra tokens={adv_tokens}, time={adv_time:.4f}s)")
        df.loc[idx, "adv_proco_answer"] = adv_final
        df.loc[idx, "adv_proco_token_cost"] = adv_tokens
        df.loc[idx, "adv_proco_time_cost"] = adv_time

    df.to_csv(output_path, index=False)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProCo pipeline compare all variants")
    parser.add_argument('--dataset_path', required=True, help="Dataset path")
    parser.add_argument('--model_key', required=True, help="Model key for Advanced_ProCo")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"Proco_outputs_{dataset_name}.csv"

    run_pipeline(dataset_path, output_path)
