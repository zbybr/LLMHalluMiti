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
import re
load_dotenv(override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


def parse_and_sort_responses(data: str):
    # pattern = re.compile(
    #     r"<response>\s*<text>(.*?)</text>\s*<probability>(.*?)</probability>\s*<confidence>(.*?)</confidence>\s*</response>",
    #     re.DOTALL
    # )
    pattern = re.compile(
        r"<response>\s*<text>(.*?)<probability>(.*?)</probability>\s*<confidence>(.*?)</confidence>\s*</response>",
        re.DOTALL
    )

    responses = []
    for text, prob, conf in pattern.findall(data):
        responses.append({
            "text": text.strip(),
            "probability": float(prob.strip()),
            "confidence": float(conf.strip())
        })

    if not responses:
        return None

    sorted_responses = sorted(
        responses,
        key=lambda x: (-x["confidence"], x["probability"])
    )

    return sorted_responses[0]["text"]


def safe_chat_call(messages, model_key, max_retries=10, base_delay=0.0):
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


def run_pipeline(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="latin-1", quoting=csv.QUOTE_ALL)
    if os.path.exists(output_path):
        print(f"Resuming from existing output file: {output_path}")
        df_out = pd.read_csv(output_path, encoding="latin-1", quoting=csv.QUOTE_ALL)
        df = df.merge(
            df_out[["Question", "final_answer", "record", "token_cost", "time_cost"]],
            on="Question",
            how="left",
            suffixes=("", "_saved")
        )
    else:
        df["final_answer"] = ""
        df["record"] = ""
        df["token_cost"] = None
        df["time_cost"] = None
    df_todo = df[df["final_answer"].isna() | (df["final_answer"].astype(str).str.strip() == "")]

    print(f"Total questions: {len(df)}")
    print(f"Already processed: {len(df) - len(df_todo)}")
    print(f"Remaining to process: {len(df_todo)}")
    for index, row in tqdm(df.iterrows(), total=len(df_todo), desc="Processing QA"):
        start = time.time()
        question = row["Question"]
        base_response = row['base_response']
        qapair = f"Question: {question}\nBase_response: {base_response}\n"
        messages = [{"role": "user", "content": prompts.VERBALIZED_SAMPLING_PROMPT + '\n' + qapair}]
        record, tokens = safe_chat_call(messages, model_key)
        final_answer = parse_and_sort_responses(record)
        end = time.time()

        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Base Response: {base_response}")
        print(f"Record: {record}")
        print(f"Final Answer: {final_answer} (tokens={tokens}, time={end - start:.4f}s)")

        # Save results into dataframe
        df.loc[index, "final_answer"] = final_answer
        df.loc[index, "record"] = record
        df.loc[index, "token_cost"] = tokens
        df.loc[index, "time_cost"] = end - start

    df.to_csv(output_path, index=False)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination Mitigation using verbalized sampling pipeline with cost tracking")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument('--model_key', type=str, required=True, help="Model key")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"{args.model_key}_vs_outputs_{dataset_name}.csv"

    run_pipeline(dataset_path, output_path, args.model_key)
