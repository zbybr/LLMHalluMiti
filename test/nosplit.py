import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import llm_prompts.prompts as prompts

load_dotenv()
GPT_MODEL_KEY = "gpt-5-chat-latest"

client = OpenAI(
    api_key="sk-wi5c6GQjiqZVC0vqDjSHZA5UIIHpmgiFgRgSyDS0PnOkJWWF",
    base_url="https://yunwu.ai/v1"
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


def run_pipeline(input_path, output_path):
    df = pd.read_csv(input_path, encoding="latin-1")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing QA"):
        question = row["Question"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        first_response = client.chat.completions.create(
            model=GPT_MODEL_KEY,
            messages=messages,
            temperature=0.0,
        ).choices[0].message.content.strip()

        messages.append({"role": "assistant", "content": first_response})

        messages.append({"role": "user", "content": recheck_prompt_nsp})

        second_response = client.chat.completions.create(
            model=GPT_MODEL_KEY,
            messages=messages,
            temperature=0.0,
        ).choices[0].message.content.strip()

        final_answer, hallucination_check = parse_rechecked_response(second_response)

        # --- Save to DataFrame ---
        df.loc[index, "base_response"] = first_response
        df.loc[index, "final_answer"] = final_answer
        df.loc[index, "hallucination_check"] = hallucination_check
        df.loc[index, "raw_rechecked_response"] = second_response

        print("===================================")
        print(f"Question: {question}")
        print(f"Base Response: {first_response}")
        print(f"Final Answer: {final_answer}")
        print(f"Hallucination Check: {hallucination_check}")

    df.to_csv(output_path, index=False)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Hallucination Mitigation pipeline.")
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"gpt_outputs_{dataset_name}_hallucination_checked_nsp.csv"

    run_pipeline(dataset_path, output_path)
