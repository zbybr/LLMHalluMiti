import os
import csv
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import llm_prompts.prompts as prompts

load_dotenv(override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


def call_llm(messages, model_key, max_retries=20, base_delay=1.0):
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
            return content.strip()
        except Exception as e:
            wait = base_delay * (attempt + 1)
            print(f"[Warning] Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying after {wait:.1f}s...")
                import time
                time.sleep(wait)
    print("[Error] Model failed after multiple retries.")
    return "ERROR: Empty or invalid model output"


def run_pipeline(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing QA"):
        correct_answer = row["Answer"]
        context = row['final_answer']
        qapair = f"Correct Answer: {correct_answer}\n\nContext: {context}"
        messages = [
            {"role": "system", "content": prompts.LLM_JUDGE_PROMPT},
            {"role": "user", "content": qapair},
                    ]
        final_answer = call_llm(messages, model_key)
        if final_answer == 'YES':
            final_answer = 'NO'
        else:
            final_answer = 'YES'

        # Logging
        print("===================================")
        print(f"Correct Answer: {correct_answer}\nContext: {context}")
        print(f"Final Judge: {final_answer}")

        # Save results into dataframe
        df.loc[index, "recheck_hallucination_llm"] = final_answer

        df.to_csv(output_path, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Judge")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    model_key = 'gpt-4o'
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"{dataset_name}_llm_judge.csv"

    run_pipeline(dataset_path, output_path, model_key)