import argparse
import csv
from tqdm import tqdm
from pathlib import Path
import prompt
import pandas as pd


def run_pipeline(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="latin-1", quoting=csv.QUOTE_ALL)

    print(f"Loaded dataset '{input_path}' with {len(df)} questions.")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        question = row["Question"]
        base_response, final_answer, tokens_used, time_used = prompt.pipeline(question, model_key)

        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Final Answer: {final_answer} (extra tokens={tokens_used}, time={time_used:.4f}s)")
        df.loc[idx, "base_response"] = base_response
        df.loc[idx, "proco_answer"] = final_answer
        df.loc[idx, "proco_token_cost"] = tokens_used
        df.loc[idx, "proco_time_cost"] = time_used

    df.to_csv(output_path, index=False)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProCo single-iteration pipeline compare all variants")
    parser.add_argument('--dataset_path', required=True, help="Dataset path")
    parser.add_argument('--model_key', required=True, help="Model key for Advanced_ProCo")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"Proco_single_outputs_{dataset_name}.csv"

    run_pipeline(dataset_path, output_path, args.model_key)
