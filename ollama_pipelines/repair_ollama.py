import argparse
import csv
import os
import random
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from tqdm import tqdm

import llm_prompts.prompts as prompts

load_dotenv(override=True)
#
#
# def parse_rechecked_response(text: str):
#     final_answer, hallucination_check = "", ""
#     for line in text.splitlines():
#         line = line.strip()
#         if line.startswith("1"):
#             final_answer = line.lstrip("1234567890. ").strip()
#         elif line.startswith("2"):
#             hallucination_check = line.lstrip("1234567890. ").strip()
#     return final_answer, hallucination_check


def safe_chat_call(messages, model_key, max_retries=10, base_delay=0.0):
    """
    Safe wrapper for ChatOllama with retries and detailed tracking.
    Returns: (content, token_cost)
    """
    llm = ChatOllama(
        model=model_key, temperature=0.0, base_url="http://localhost:11439", timeout=60
    )

    for attempt in range(max_retries):
        try:
            # Invoke the model
            response = llm.invoke(messages)
            content = response.content

            if not content or not content.strip():
                raise ValueError("Empty or null response from model")

            content = content.strip()

            print(response)
            # Get token usage from response metadata
            tokens = 0
            if hasattr(response, "response_metadata") and response.response_metadata:
                prompt_tokens = response.response_metadata.get("prompt_eval_count", 0)
                completion_tokens = response.response_metadata.get("eval_count", 0)
                tokens = prompt_tokens + completion_tokens

            return content, tokens

        except Exception as e:
            wait = base_delay * (attempt + 1) + random.uniform(0, 1)
            print(f"[Warning] Attempt {attempt+1}/{max_retries} failed: {e}")
            print(f"Retrying after {wait:.1f}s ...")
            time.sleep(wait)

    print("[Error] Model failed after multiple retries.")
    return "ERROR: Empty or invalid model output", 0


def run_pipeline(input_path, output_path, model_key):
    # Load existing output if it exists, otherwise load input dataset
    if os.path.exists(output_path):
        print(f"Loading existing output from {output_path}")
        df = pd.read_csv(output_path, encoding="latin-1", quoting=csv.QUOTE_ALL)
    else:
        print(f"Loading input dataset from {input_path}")
        df = pd.read_csv(input_path, encoding="latin-1", quoting=csv.QUOTE_ALL)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing QA"):
        # Skip if final_answer already exists and is not empty
        if (
            "final_answer" in df.columns
            and pd.notna(row.get("final_answer"))
            and str(row.get("final_answer")).strip()
        ):
            print(f"Skipping index {index} as final_answer already exists.")
            continue

        question = row["Question"]
        print(f"Processing index {index}: {question}")
        prompt = (
            f"Answer the following question accurately with one-complete sentence."
            f"If the question is subjective, answer 'I have no idea.':\n"
            f"Question: {question}\n"
        )
        base_messages = [
            {"role": "user", "content": prompt},
        ]
        base_response = safe_chat_call(base_messages, model_key)[0]
        print(f"Base response: {base_response}")
        qapair = f"Question: {question}\n\nBase_response: {base_response}"
        start = time.time()
        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": qapair},
        ]
        print("Messages to model:")
        print(messages)
        final_answer, tokens = safe_chat_call(messages, model_key)
        # record, tokens = safe_chat_call(messages, model_key)
        # final_answer, hallucination_check = parse_rechecked_response(record)
        end = time.time()

        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Base Response: {base_response}")
        print(
            f"Final Answer: {final_answer} (tokens={tokens}, time={end - start:.4f}s)"
        )
        # print(f"Hallucination Check: {hallucination_check}")

        # Save results into dataframe
        df.loc[index, "final_answer"] = final_answer
        # df.loc[index, "hallucination_check"] = hallucination_check
        # df.loc[index, "raw_rechecked_response"] = record
        df.loc[index, "token_cost"] = tokens
        df.loc[index, "time_cost"] = end - start

        # Save after each row to preserve progress
        df.to_csv(output_path, index=False)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hallucination Mitigation pipeline with cost tracking"
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--model_key", type=str, required=True, help="Model key")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"{args.model_key}_outputs_{dataset_name}.csv"

    run_pipeline(dataset_path, output_path, args.model_key)
