# Backward-compatible CLI: shared implementation, original imports/functions remain below.
if __name__ == "__main__":
    from pathlib import Path as _Path
    import sys as _sys

    _model_dir = _Path(__file__).resolve().parent.parent
    if str(_model_dir) not in _sys.path:
        _sys.path.insert(0, str(_model_dir))
    from _experiment import run as _run_shared

    _run_shared("leetcode.generate_base_responses", __file__)
    raise SystemExit


import os
import csv
import argparse
from pathlib import Path
from openai import OpenAI
from pathlib import Path as _ConfigPath
import sys as _config_sys

_config_root = next(
    parent for parent in _ConfigPath(__file__).resolve().parents
    if (parent / "common" / "config.py").is_file()
)
if str(_config_root) not in _config_sys.path:
    _config_sys.path.insert(0, str(_config_root))
from common.config import ROOT_ENV, load_root_env, read_env
import pandas as pd
from tqdm import tqdm

load_root_env()
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
                temperature=0.1,
                extra_body={"enable_thinking": False}
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
    model_key = 'qwen3-32b'
    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"{model_key}_{dataset_name}_responses.csv"
    main(dataset_path, output_path, model_key)
