import argparse
import random
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import os
import csv
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '..', '.env')
load_dotenv(dotenv_path=ENV_PATH, override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


def safe_chat_call(messages, model_key, max_retries=3, base_delay=2.0):
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
            return content

        except Exception as e:
            wait = base_delay * (attempt + 1) + random.uniform(0, 1)
            print(f"[Warning] Attempt {attempt+1}/{max_retries} failed: {e}")
            print(f"Retrying after {wait:.1f}s ...")
            time.sleep(wait)

    print("[Error] Model failed after multiple retries.")
    return "ERROR: Empty or invalid model output"


def run_pipeline(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="latin-1", quoting=csv.QUOTE_ALL)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing QA"):
        question = row["Question"]
        base_response = row['base_response']
        prompt = f"""You are given an answer sentence to a question. Your task is to extract or transform the answer 
        into a single concise entity (e.g., a person, place, organization, object, or concept). If the answer does 
        not provide factual content (e.g., "I have no idea", "I have no comment"),output the entity as "Unknown".
        Question: {question}
        Answer: {base_response}
        The single entity is:
        """
        messages = [
            {"role": "assistant", "content": prompt},
        ]
        entity = safe_chat_call(messages, model_key)

        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Base Response: {base_response}")
        print(f"Entity: {entity}")

        # Save results into dataframe
        df.loc[index, "entity"] = entity

    df.to_csv(output_path, index=False)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity generation pipeline")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument('--model_key', type=str, required=True, help="Model key")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"{args.model_key}_outputs_{dataset_name}_entity.csv"

    run_pipeline(dataset_path, output_path, args.model_key)
