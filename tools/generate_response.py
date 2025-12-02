import random
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


def call_llm(prompt, model_key, max_retries=10, base_delay=3.0):
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


def generate_response(question, correct_answer, model_key):
    prompt = (
        f"Answer the following question accurately with one sentence based on first sentence in correct answer. "
        f"Additionally, for the subjective question, 'I have no idea.' is considered as correct answer:\n"
        f"Question: {question}\n"
        f"Correct Answer: {correct_answer}"
    )
    return call_llm(prompt, model_key)


def main(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="latin-1", quoting=csv.QUOTE_ALL)

    base_responses = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing QA"):
        question = row["Question"]
        answer = row["Answer"]
        response= generate_response(question, answer, model_key)
        base_responses.append(response)

        print("===================================")
        print(f"Question: {question}")
        print(f"Generated Response: {response}")

    df["base_response"] = base_responses

    # Save new dataset
    df.to_csv(output_path, index=False)
    print(f"Dataset saved as {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Response Dataset Generation")
    parser.add_argument('--dataset_path', type=str, required=True, help="Dataset path")
    parser.add_argument('--model_key', type=str, required=True, help="Model key")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"{dataset_name}_responses.csv"
    main(dataset_path, output_path, args.model_key)
