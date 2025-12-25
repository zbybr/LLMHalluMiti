import argparse
import csv
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import llm_prompts.prompts as prompts
from ollama_pipelines.ollama_util import safe_chat_call


def parse_rechecked_response(text: str):
    final_answer, hallucination_check = "", ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("1"):
            final_answer = line.lstrip("1234567890. ").strip()
        elif line.startswith("2"):
            hallucination_check = line.lstrip("1234567890. ").strip()
    return final_answer, hallucination_check


def run_pipeline(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="iso-8859-1", quoting=csv.QUOTE_ALL).sample(1)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing QA"):
        question = row["Question"]
        base_response = row["base_response"]
        qapair = f"Question: {question}\n\nBase_response: {base_response}"
        start = time.time()
        messages = [{"role": "user", "content": qapair + "\n" + prompts.COT_PROMPT}]
        final_answer, tokens = safe_chat_call(messages, model_key)

        end = time.time()
        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Base Response: {base_response}")
        print(
            f"Final Answer: {final_answer} (tokens={tokens}, time={end - start:.4f}s)"
        )

        # Save results into dataframe
        df.loc[index, "final_answer"] = final_answer
        df.loc[index, "token_cost"] = tokens
        df.loc[index, "time_cost"] = end - start

    df.to_csv(output_path, encoding="iso-8859-1", index=False)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COT pipeline with cost tracking")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--model_key", type=str, required=True, help="Model key")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    model_key_formatted = args.model_key.replace(":", "_")
    output_path = f"./outputs/cot/{model_key_formatted}_cot_outputs_{dataset_name}.csv"

    run_pipeline(dataset_path, output_path, args.model_key)
