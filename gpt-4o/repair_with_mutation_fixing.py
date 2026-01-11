import argparse
import csv
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import llm_prompts.prompts as prompts
import utils

load_dotenv(override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)


def extract_mutations(text: str):
    sentences = re.findall(r"(?:\d+[\.\)]\s*)([^0-9\n]+(?:[^\n]*))", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def safe_chat_call(messages, model_key, max_retries=20, base_delay=0.0):
    """
    Safe wrapper for OpenAI chat completion with retries and detailed tracking.
    Returns: (content, token_cost)
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_key,
                messages=messages,
                temperature=0.0,
            )

            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty or null response from model")

            content = content.strip()
            utils.check_string(content)

            input_text = "\n".join(m["content"] for m in messages if "content" in m)
            tokens = utils.num_tokens_from_string(
                input_text
            ) + utils.num_tokens_from_string(content)

            return content, tokens

        except Exception as e:
            wait = base_delay * (attempt + 1) + random.uniform(0, 1)
            print(f"[Warning] Attempt {attempt+1}/{max_retries} failed: {e}")
            print(f"Retrying after {wait:.1f}s ...")
            time.sleep(wait)

    print("[Error] Model failed after multiple retries.")
    return "ERROR: Empty or invalid model output", 0


def run_pipeline(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    init_cols = [
        "final_answer_cs",
        "mutation_list"
    ]
    for c in init_cols:
        if c not in df.columns:
            df[c] = pd.NA
    if os.path.exists(output_path):
        print(f"Resuming from existing output file: {output_path}")
        df_out = pd.read_csv(output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        merge_cols = [c for c in df_out.columns if c in df.columns or c not in df.columns]
        df = df.merge(df_out[merge_cols], on="Question", how="left", suffixes=("", "_saved"))
    else:
        for c in init_cols:
            saved = c + "_saved"
            if saved in df.columns:
                if df[c].dtype == "object":
                    base_missing = df[c].isna() | (df[c].astype(str).str.strip() == "")
                    df.loc[base_missing, c] = df.loc[base_missing, saved]
                else:
                    df[c] = df[c].combine_first(df[saved])
                df.drop(columns=[saved], inplace=True)

    condition = (df["final_answer_cs"].isna() | (df["final_answer_cs"].astype(str).str.strip() == ""))
    df_todo = df[condition]

    print(f"Total questions: {len(df)}")
    print(f"Already processed: {len(df) - len(df_todo)}")
    print(f"Remaining to process: {len(df_todo)}")
    for index, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Processing QA"):
        record = []
        question = row["Question"]
        mutation_list = [s.strip() for s in str(row["mutation_list"]).splitlines() if s.strip()]
        for mutation in mutation_list:
            qapair = f"Question: {question}\nBase_response: {mutation}"
            messages = [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": qapair},
            ]
            answer, _tokens = safe_chat_call(messages, model_key)
            record.append(answer.strip())
        record_str = "\n".join(record)

        # confidence score
        messages = [
            {"role": "system", "content": prompts.CONFIDENCE_SCORE_PROMPT},
            {"role": "user","content": f"Question: {question}\nAnswers: {record_str}"},
        ]
        confidence_score_result, _tokens = safe_chat_call(messages, model_key)
        messages = [
            {"role": "system", "content": prompts.REFINE_PROMPT},
            {"role": "user", "content": f"{confidence_score_result}"},
        ]
        final_answer_cs, _tokens = safe_chat_call(messages, model_key)

        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Final Answer by Confidence Score: {final_answer_cs}")

        # Save results into dataframe
        df.loc[index, "final_answer_cs"] = final_answer_cs

        df.to_csv(output_path, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination Mitigation using mutations pipeline with cost tracking")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    # parser.add_argument('--model_key', type=str, required=True, help="Model key")
    args = parser.parse_args()
    model_key = 'gpt-4o'
    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"./outputs/output.csv"

    run_pipeline(dataset_path, output_path, model_key)
