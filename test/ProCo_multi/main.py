import argparse
import csv
import warnings
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import prompt
import json

warnings.filterwarnings('ignore')
prompt_strategy = 'Generate-Read-Refine'
encoding_name = "cl100k_base"
MAX_ITERATION = 3


def run_pipeline(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="latin-1", quoting=csv.QUOTE_ALL)

    print(f"Loaded dataset '{input_path}' with {len(df)} questions.")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        question = row["Question"]
        base_response = row['base_response']
        entity = row['entity']
        final_answer, process_record, tokens_used, time_used = prompt.pipeline(question, entity, {}, model_key, MAX_ITERATION)

        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Base Response: {base_response}, Final Answer: {final_answer} (tokens={tokens_used}, time={time_used:.4f}s)")
        df.loc[idx, "proco_answer"] = final_answer
        df.loc[idx, "proco_record"] = json.dumps(process_record, ensure_ascii=False)
        df.loc[idx, "proco_token_cost"] = tokens_used
        df.loc[idx, "proco_time_cost"] = time_used

    df.to_csv(output_path, index=False)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProCo single-iteration pipeline compare all variants")
    parser.add_argument('--dataset_path', type=str, required=True, help="Dataset path")
    parser.add_argument('--model_key', type=str, required=True, help="Model key for ProCo")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"{args.model_key}_proco_multi_outputs_{dataset_name}.csv"

    run_pipeline(dataset_path, output_path, args.model_key)
