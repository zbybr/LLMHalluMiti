import argparse
import csv
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import llm_prompts.prompts as prompts
from ollama_util import safe_chat_call

load_dotenv(override=True)


def extract_mutations(text: str):
    sentences = re.findall(r"(?:\d+[\.\)]\s*)([^0-9\n]+(?:[^\n]*))", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def run_pipeline(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="latin-1", quoting=csv.QUOTE_ALL)
    if os.path.exists(output_path):
        print(f"Resuming from existing output file: {output_path}")
        df_out = pd.read_csv(output_path, encoding="latin-1", quoting=csv.QUOTE_ALL)
        df = df.merge(
            df_out[
                [
                    "Question",
                    "final_answer",
                    "mutation_list",
                    "answer_list",
                    "token_cost",
                    "time_cost",
                ]
            ],
            on="Question",
            how="left",
            suffixes=("", "_saved"),
        )
    else:
        df["final_answer"] = ""
        df["mutation_list"] = ""
        df["answer_list"] = ""
        df["token_cost"] = None
        df["time_cost"] = None
    df_todo = df[
        df["final_answer"].isna() | (df["final_answer"].astype(str).str.strip() == "")
    ]

    print(f"Total questions: {len(df)}")
    print(f"Already processed: {len(df) - len(df_todo)}")
    print(f"Remaining to process: {len(df_todo)}")
    for index, row in tqdm(df.iterrows(), total=len(df_todo), desc="Processing QA"):
        start = time.time()
        record = []
        question = row["Question"]
        base_response = row["base_response"]
        qapair = f"Question: {question}\nBase_response: {base_response}"
        messages = [
            {"role": "system", "content": prompts.MUTATION_PROMPT},
            {"role": "user", "content": qapair},
        ]
        mutations, tokens = safe_chat_call(messages, model_key)
        mutation_list = extract_mutations(mutations)
        mutation_list.append(base_response)
        for mutation in mutation_list:
            qapair = f"Question: {question}\nBase_response: {mutation}"
            messages = [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": qapair},
            ]
            answer, _tokens = safe_chat_call(messages, model_key)
            tokens += _tokens
            record.append(answer.strip())

        record_str = "\n".join(record)
        messages = [
            {"role": "system", "content": prompts.JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\nAnswers: {record_str}\n",
            },
        ]
        final_answer, _tokens = safe_chat_call(messages, model_key)
        tokens += _tokens
        end = time.time()
        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Base Response: {base_response}")
        print(
            f"Final Answer: {final_answer} (tokens={tokens}, time={end - start:.4f}s)"
        )
        mutation_list_str = "\n".join(mutation_list)
        # Save results into dataframe
        df.loc[index, "final_answer"] = final_answer
        df.loc[index, "mutation_list"] = mutation_list_str
        df.loc[index, "answer_list"] = record_str
        df.loc[index, "token_cost"] = tokens
        df.loc[index, "time_cost"] = end - start

    df.to_csv(output_path, index=False)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hallucination Mitigation using mutations pipeline with cost tracking"
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--model_key", type=str, required=True, help="Model key")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"{args.model_key}_mutation_outputs_{dataset_name}.csv"

    run_pipeline(dataset_path, output_path, args.model_key)
