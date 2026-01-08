import argparse
import random
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import llm_prompts.prompts as prompts
import os
import csv
import time
import utils

load_dotenv(override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


def parse_rechecked_response(text: str):
    final_answer, hallucination_check = "", ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("1"):
            final_answer = line.lstrip("1234567890. ").strip()
        elif line.startswith("2"):
            hallucination_check = line.lstrip("1234567890. ").strip()
    return final_answer, hallucination_check


def safe_chat_call(messages, model_key, max_retries=20, base_delay=0.0):
    """
    Safe wrapper for OpenAI chat completion with retries and detailed tracking.
    Returns: (content, token_cost, time_cost)
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
            tokens = utils.num_tokens_from_string(input_text) + utils.num_tokens_from_string(content)

            return content, tokens

        except Exception as e:
            wait = base_delay * (attempt + 1) + random.uniform(0, 1)
            print(f"[Warning] Attempt {attempt+1}/{max_retries} failed: {e}")
            print(f"Retrying after {wait:.1f}s ...")
            time.sleep(wait)

    print("[Error] Model failed after multiple retries.")
    return "ERROR: Empty or invalid model output", 0


def run_pipeline(input_path, output_path, model_key='gemini-3-pro-preview-11-2025'):
    df = pd.read_csv(input_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing QA"):
        question = row["Question"]
        base_response = row['base_response']
        qapair = f"Question: {question}\n\nBase_response: {base_response}"
        start = time.time()
        messages = [{"role": "system", "content": qapair + '\n' + prompts.COT_PROMPT}]
        final_answer, tokens = safe_chat_call(messages, model_key)

        end = time.time()
        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Base Response: {base_response}")
        print(f"Final Answer: {final_answer} (tokens={tokens}, time={end - start:.4f}s)")

        # Save results into dataframe
        df.loc[index, "final_answer"] = final_answer
        df.loc[index, "token_cost"] = tokens
        df.loc[index, "time_cost"] = end - start

        df.to_csv(output_path, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COT pipeline with cost tracking")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    # parser.add_argument('--model_key', type=str, required=True, help="Model key")
    args = parser.parse_args()

    model_key = "gemini-3-pro-preview-11-2025"
    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"./outputs/cot/gemini_cot_outputs_{dataset_name}.csv"

    run_pipeline(dataset_path, output_path, model_key)
