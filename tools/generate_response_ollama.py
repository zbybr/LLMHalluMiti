import argparse
import csv
import os
import random
import time
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama
from tqdm import tqdm


def safe_chat_call(messages, model_key, max_retries=5, base_delay=0.0):
    """
    Safe wrapper for ChatOllama with retries and detailed tracking.
    Returns: (content, token_cost)
    """
    llm = ChatOllama(
        model=model_key,
        # temperature=0.0,
        base_url="http://localhost:11439",
        timeout=30,
        cache=False,
        num_predict=2048,
    )

    for attempt in range(max_retries):
        try:
            # Invoke the model
            response = llm.invoke(messages)
            content = response.content

            if not content or not content.strip():
                raise ValueError("Empty or null response from model")

            content = content.strip()

            # Get token usage from response metadata
            tokens = 0
            if hasattr(response, "response_metadata") and response.response_metadata:
                prompt_tokens = response.response_metadata.get("prompt_eval_count", 0)
                completion_tokens = response.response_metadata.get("eval_count", 0)
                tokens = prompt_tokens + completion_tokens

            return content, tokens

        except Exception as e:
            wait = base_delay * (attempt + 1) + random.uniform(0, 1)
            print(f"[Warning] Attempt {attempt+1}/{max_retries} failed: {e}")
            print(f"Retrying after {wait:.1f}s ...")
            time.sleep(wait)

    print("[Error] Model failed after multiple retries.")
    return "ERROR: Empty or invalid model output", 0


def generate_response(question, model_key):
    prompt = f"""Answer the following question accurately with only one sentence.
    Do not explain.
    Do NOT add anything else.
    If the question is subjective, answer 'I have no idea.'

    Question:
    {question}
    """.strip()
    # messages = [{"role": "user", "content": prompt}]
    return safe_chat_call(prompt, model_key)


def main(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="utf-8", quoting=csv.QUOTE_ALL)

    base_responses = []
    base_answers = []
    base_questions = []

    # Load existing outputs if available
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, encoding="utf-8", quoting=csv.QUOTE_ALL)
        base_responses = existing_df["base_response"].tolist()
        base_answers = existing_df["Answer"].tolist()
        base_questions = existing_df["Question"].tolist()
        start_index = len(base_responses)
        print(f"Resuming from index {start_index}")
        df = df.iloc[start_index:]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing QA"):
        question = row["Question"]
        answer = row["Answer"]
        response, _ = generate_response(question, model_key)
        base_responses.append(response)
        base_answers.append(answer)
        base_questions.append(question)

        print("===================================")
        print(f"Question: {question}")
        print(f"Generated Response: {response}")

        output_df = pd.DataFrame(
            {
                "Question": base_questions,
                "Answer": base_answers,
                "base_response": base_responses,
            }
        )
        output_df.to_csv(
            output_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL
        )

        print(f"Dataset saved as {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Response Dataset Generation")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--model_key", type=str, required=True, help="Model key")
    args = parser.parse_args()
    model_key = args.model_key
    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"{model_key}_{dataset_name}_responses.csv"
    main(dataset_path, output_path, model_key)

# safe_chat_call("Who are you?", "gpt-oss:20b")
