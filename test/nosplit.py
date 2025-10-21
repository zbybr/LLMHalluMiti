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
GPT_MODEL_KEY = "gpt-5-2025-08-07"
# client = OpenAI(
#     api_key="sk-wi5c6GQjiqZVC0vqDjSHZA5UIIHpmgiFgRgSyDS0PnOkJWWF",
#     base_url="https://yunwu.ai/v1"
# )
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
system_prompt = prompts.SYSTEM_PROMPT
recheck_prompt_nsp = prompts.RECHECK_PROMPT_NOSPLIT


def parse_rechecked_response(text: str):
    final_answer, hallucination_check = "", ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("1"):
            final_answer = line.lstrip("1234567890. ").strip()
        elif line.startswith("2"):
            hallucination_check = line.lstrip("1234567890. ").strip()
    return final_answer, hallucination_check


def safe_chat_call(messages, model_key, max_retries=3, base_delay=2.0):
    """
    Safe wrapper for OpenAI chat completion with retries and detailed tracking.
    Returns: (content, token_cost, time_cost)
    """
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            response_obj = client.chat.completions.create(
                model=model_key,
                messages=messages,
                temperature=0.0,
            )
            end = time.perf_counter()

            content = response_obj.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty or null response from model")

            content = content.strip()
            utils.check_string(content)

            input_text = "\n".join(m["content"] for m in messages if "content" in m)
            tokens = utils.num_tokens_from_string(input_text) + utils.num_tokens_from_string(content)

            time_cost = end - start
            return content, tokens, time_cost

        except Exception as e:
            wait = base_delay * (attempt + 1) + random.uniform(0, 1)
            print(f"[Warning] Attempt {attempt+1}/{max_retries} failed: {e}")
            print(f"Retrying after {wait:.1f}s ...")
            time.sleep(wait)

    print("[Error] Model failed after multiple retries.")
    return "ERROR: Empty or invalid model output", 0, 0.0


def run_pipeline(input_path, output_path):
    df = pd.read_csv(input_path, encoding="latin-1", quoting=csv.QUOTE_ALL)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing QA"):
        question = row["Question"]
        start = time.time()
        # Step 1: Base response with SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        first_response, first_tokens, first_time = safe_chat_call(messages, GPT_MODEL_KEY)

        # Step 2: Re-check with RECHECK_PROMPT_NOSPLIT
        messages.append({"role": "assistant", "content": first_response})
        messages.append({"role": "user", "content": prompts.RECHECK_PROMPT_NOSPLIT})
        second_response, second_tokens, second_time = safe_chat_call(messages, GPT_MODEL_KEY)

        # Step 3: Parse hallucination mitigation output
        final_answer, hallucination_check = parse_rechecked_response(second_response)

        end = time.time()
        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Base Response: {first_response} (tokens={first_tokens}, time={first_time:.4f}s)")
        print(f"Final Answer: {final_answer} (tokens={second_tokens}, time={second_time:.4f}s)")
        print(f"Hallucination Check: {hallucination_check}")

        # Save results into dataframe
        df.loc[index, "base_response"] = first_response
        df.loc[index, "final_answer"] = final_answer
        df.loc[index, "hallucination_check"] = hallucination_check
        df.loc[index, "raw_rechecked_response"] = second_response
        df.loc[index, "base_token_cost"] = first_tokens
        df.loc[index, "base_time_cost"] = first_time
        df.loc[index, "recheck_token_cost"] = second_tokens
        df.loc[index, "recheck_time_cost"] = second_time
        df.loc[index, "token_cost"] = first_tokens + second_tokens
        df.loc[index, "time_cost"] = end - start

    df.to_csv(output_path, index=False)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Hallucination Mitigation pipeline with cost tracking")
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"{GPT_MODEL_KEY}_outputs_{dataset_name}_hallucination_checked_nsp.csv"

    run_pipeline(dataset_path, output_path)
