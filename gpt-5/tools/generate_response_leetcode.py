import os
import csv
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

load_dotenv(override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


def call_llm(prompt, model_key, max_retries=20, base_delay=2.0):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_key,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9
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


def generate_response(question, model_key):
    return call_llm(question, model_key)


def main(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    base_responses = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing QA"):
        question = row["query"]
        response= generate_response(question, model_key)
        base_responses.append(response)

        print("===================================")
        print(f"Question: {question}")
        print(f"Generated Response: {response}")

    df["base_response"] = base_responses

    # Save new dataset
    df.to_csv(output_path, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)
    print(f"Dataset saved as {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Response Dataset Generation")
    parser.add_argument('--dataset_path', type=str, required=True, help="Dataset path")
    # parser.add_argument('--model_key', type=str, required=True, help="Model key")
    args = parser.parse_args()
    model_key = 'gpt-5'
    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"{model_key}_{dataset_name}_responses.csv"
    main(dataset_path, output_path, model_key)
