import argparse
import csv
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import llm_prompts.prompts as prompts
import utils

load_dotenv(override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)


def extract_mutations(text: str):
    sentences = re.findall(r"(?:\d+[\.\)]\s*)([^0-9\n]+(?:[^\n]*))", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def safe_chat_call(messages, model_key, max_retries=10, base_delay=0.0):
    """
    Safe wrapper for OpenAI chat completion with retries and detailed tracking.
    Returns: (content, token_cost)
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
            tokens = utils.num_tokens_from_string(
                input_text
            ) + utils.num_tokens_from_string(content)

            return content, tokens

        except Exception as e:
            wait = base_delay * (attempt + 1) + random.uniform(0, 1)
            print(f"[Warning] Attempt {attempt+1}/{max_retries} failed: {e}")
            print(f"Retrying after {wait:.1f}s ...")
            time.sleep(wait)

    print("[Error] Model failed after multiple retries.")
    return "ERROR: Empty or invalid model output", 0


def run_pipeline(input_path, output_path, model_key):
    df = pd.read_csv(input_path, encoding="iso-8859-1", quoting=csv.QUOTE_ALL)
    if os.path.exists(output_path):
        print(f"Resuming from existing output file: {output_path}")
        df_out = pd.read_csv(output_path, encoding="iso-8859-1", quoting=csv.QUOTE_ALL)
        merge_cols = [
            c for c in df_out.columns if c in df.columns or c not in df.columns
        ]
        df = df.merge(
            df_out[merge_cols], on="Question", how="left", suffixes=("", "_saved")
        )
    else:
        print("No existing output file found. Initializing columns...")
    init_cols = [
        "final_answer_mv",
        "token_cost_mv",
        "time_cost_mv",
        "final_answer_cs",
        "token_cost_cs",
        "time_cost_cs",
        "final_answer_ra",
        "token_cost_ra",
        "time_cost_ra",
        "mutation_list",
        "answer_list",
    ]
    for col in init_cols:
        if col not in df.columns:
            if "token_cost" in col or "time_cost" in col:
                df[col] = None
            else:
                df[col] = ""

    condition = (
        (
            df["final_answer_mv"].isna()
            | (df["final_answer_mv"].astype(str).str.strip() == "")
        )
        | (
            df["final_answer_cs"].isna()
            | (df["final_answer_cs"].astype(str).str.strip() == "")
        )
        | (
            df["final_answer_ra"].isna()
            | (df["final_answer_ra"].astype(str).str.strip() == "")
        )
    )
    df_todo = df[condition]

    print(f"Total questions: {len(df)}")
    print(f"Already processed: {len(df) - len(df_todo)}")
    print(f"Remaining to process: {len(df_todo)}")
    for index, row in tqdm(df.iterrows(), total=len(df_todo), desc="Processing QA"):
        start = time.time()
        record = []
        question = row["Question"]
        base_response = row["base_response"]
        qapair = f"Question: {question}\nBase_response: {base_response}"
        messages = [
            {"role": "system", "content": prompts.MUTATION_PROMPT},
            {"role": "user", "content": qapair},
        ]
        mutations, tokens = safe_chat_call(messages, model_key)
        mutation_list = extract_mutations(mutations)
        mutation_list.append(base_response)
        for mutation in mutation_list:
            qapair = f"Question: {question}\nBase_response: {mutation}"
            messages = [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": qapair},
            ]
            answer, _tokens = safe_chat_call(messages, model_key)
            tokens += _tokens
            record.append(answer.strip())
        end = time.time()
        time_mu = end - start
        record_str = "\n".join(record)

        # majority voting
        start_mv = time.time()
        messages = [
            {"role": "system", "content": prompts.VOTING_PROMPT},
            {"role": "user", "content": f"Question: {question}\nAnswers: {record_str}"},
        ]
        final_answer_mv, _tokens = safe_chat_call(messages, model_key)
        tokens_mv = tokens + _tokens
        end_mv = time.time()

        # confidence score
        start_cs = time.time()
        messages = [
            {"role": "system", "content": prompts.CONFIDENCE_SCORE_PROMPT},
            {"role": "user","content": f"Question: {question}\nAnswers: {record_str}"},
        ]
        confidence_score_result, _tokens = safe_chat_call(messages, model_key)
        tokens_cs = tokens + _tokens
        messages = [
            {"role": "system", "content": prompts.REFINE_PROMPT},
            {"role": "user", "content": f"{confidence_score_result}"},
        ]
        final_answer_cs, _tokens = safe_chat_call(messages, model_key)
        tokens_cs += _tokens
        end_cs = time.time()

        # ranking
        start_ra = time.time()
        messages = [
            {"role": "system", "content": prompts.RANKING_PROMPT},
            {"role": "user", "content": f"Question: {question}\nAnswers: {record_str}"},
        ]
        ranking_result, _tokens = safe_chat_call(messages, model_key)
        tokens_ra = tokens + _tokens
        messages = [
            {"role": "system", "content": prompts.REFINE_PROMPT},
            {"role": "user", "content": f"{ranking_result}"},
        ]
        final_answer_ra, _tokens = safe_chat_call(messages, model_key)
        tokens_ra += _tokens
        end_ra = time.time()

        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Base Response: {base_response}")
        print(
            f"Final Answer by Majority Voting: {final_answer_mv} (tokens={tokens_mv}, time={time_mu + end_mv - start_mv:.4f}s)"
        )
        print(
            f"Final Answer by Confidence Score: {final_answer_cs} (tokens={tokens_cs}, time={time_mu + end_cs - start_cs:.4f}s)"
        )
        print(
            f"Final Answer by Ranking: {final_answer_ra} (tokens={tokens_ra}, time={time_mu + end_ra - start_ra:.4f}s)"
        )
        mutation_list_str = "\n".join(mutation_list)

        # Save results into dataframe
        df.loc[index, "mutation_list"] = mutation_list_str
        df.loc[index, "answer_list"] = record_str
        df.loc[index, "final_answer_mv"] = final_answer_mv
        df.loc[index, "token_cost_mv"] = tokens_mv
        df.loc[index, "time_cost_mv"] = time_mu + end_mv - start_mv
        df.loc[index, "final_answer_cs"] = final_answer_cs
        df.loc[index, "token_cost_cs"] = tokens_cs
        df.loc[index, "time_cost_cs"] = time_mu + end_cs - start_cs
        df.loc[index, "final_answer_ra"] = final_answer_ra
        df.loc[index, "token_cost_ra"] = tokens_ra
        df.loc[index, "time_cost_ra"] = time_mu + end_ra - start_ra

    df.to_csv(output_path, index=False)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hallucination Mitigation using mutations pipeline with cost tracking"
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--model_key", type=str, required=True, help="Model key")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"./outputs/{args.model_key}_mutation_outputs_{dataset_name}.csv"

    run_pipeline(dataset_path, output_path, args.model_key)
