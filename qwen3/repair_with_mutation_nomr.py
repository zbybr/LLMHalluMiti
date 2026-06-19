"""
repair_with_mutation_nomr.py
============================
ABLATION of repair_with_mutation.py for the MutRepair QA pipeline.

Only Stage 1 (mutation generation) is changed:

  repair_with_mutation.py   (original):
      ONE LLM call with MUTATION_PROMPT that explicitly requests Metamorphic-
      Relation-guided rewrites (synonym/antonym substitution, structural
      change, condition add/remove, viewpoint shift, cause-effect swap,
      commonsense twist), returning 5 mutations as a numbered list.

  repair_with_mutation_nomr.py  (this file):
      NO Metamorphic Relations. Mutations are produced by the naive multi-round
      method: the model is asked, in `n_mutations` INDEPENDENT calls, to simply
      restate the base response in its own words (MUTATION_PROMPT_NOMR, which
      names no MR strategy). Diversity comes only from independent sampling.

Everything else — Stage 2 repair, and Stage 3 aggregation — is identical to
repair_with_mutation.py EXCEPT that Stage 3 keeps only Pairwise Ranking (RA);
Majority Voting (MV) and Confidence Score (CS) are removed. Any metric
difference vs. the MR pipeline is therefore attributable solely to the removal
of Metamorphic Relations from the mutation step.

Input CSV columns : Question, Answer, base_response   (same as the original)
Output columns    : final_answer_ra, token_cost_ra, time_cost_ra,
                    mutation_list, answer_list
"""

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


def safe_chat_call(messages, model_key, max_retries=20, base_delay=0.0,
                   temperature=0.0):
    """
    Safe wrapper for OpenAI chat completion with retries and token tracking.
    Returns: (content, token_cost)

    A `temperature` argument is added (defaulting to 0.0 to match the original
    safe_chat_call) so the multi-round mutation step can sample with a higher
    temperature and obtain diverse restatements across independent calls.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_key,
                messages=messages,
                temperature=temperature,
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


def generate_mutations_nomr(question, base_response, model_key, n_mutations):
    """
    Stage 1 (NO Metamorphic Relations) — naive multi-round mutation.

    Instead of one MR-guided call that returns a numbered list, this issues
    `n_mutations` INDEPENDENT single-sentence restatement calls. No MR strategy
    is named; diversity arises only from independent sampling (temperature 0.9).

    Returns: (mutation_list, total_tokens)
             mutation_list has length n_mutations + 1 (mutations + base_response),
             matching the original pipeline's pool size.
    """
    mutation_list = []
    total_tokens = 0
    qapair = f"Question: {question}\nBase_response: {base_response}"
    for _ in range(n_mutations):
        messages = [
            {"role": "system", "content": prompts.MUTATION_PROMPT_NOMR},
            {"role": "user", "content": qapair},
        ]
        mutation, _tokens = safe_chat_call(
            messages, model_key, temperature=0.9
        )
        total_tokens += _tokens
        mutation_list.append(mutation.strip())
    mutation_list.append(base_response)   # keep the original base in the pool
    return mutation_list, total_tokens


def run_pipeline(input_path, output_path, model_key, n_mutations=5):
    df = pd.read_csv(input_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    init_cols = [
        "final_answer_ra", "token_cost_ra", "time_cost_ra",
        "mutation_list", "answer_list"
    ]
    for c in init_cols:
        if c not in df.columns:
            df[c] = pd.NA

    if os.path.exists(output_path):
        print(f"Resuming from existing output file: {output_path}")
        df_out = pd.read_csv(output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        merge_cols = ["Question"] + [c for c in init_cols if c in df_out.columns]
        df = df.merge(df_out[merge_cols], on="Question", how="left", suffixes=("", "_saved"))
        for c in init_cols:
            saved = c + "_saved"
            if saved in df.columns:
                base_missing = df[c].isna() | (df[c].astype(str).str.strip() == "")
                df.loc[base_missing, c] = df.loc[base_missing, saved]
                df.drop(columns=[saved], inplace=True)

    condition = (
        df["final_answer_ra"].isna() | (df["final_answer_ra"].astype(str).str.strip() == "")
    )
    df_todo = df[condition]

    print(f"Total questions: {len(df)}")
    print(f"Already processed: {len(df) - len(df_todo)}")
    print(f"Remaining to process: {len(df_todo)}")
    for index, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Processing QA (no-MR)"):
        start = time.time()
        question = row["Question"]
        base_response = row["base_response"]

        # ── Stage 1 (NO MR): naive multi-round mutation ──────────────────────
        mutation_list, tokens = generate_mutations_nomr(
            question, base_response, model_key, n_mutations
        )

        # ── Stage 2: repair each mutation (identical to the original) ─────────
        record = []
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

        # ── Stage 3: aggregation — Pairwise Ranking ONLY ─────────────────────
        # (Majority Voting and Confidence Score removed for this ablation.)
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
            f"Final Answer by Ranking: {final_answer_ra} (tokens={tokens_ra}, time={time_mu + end_ra - start_ra:.4f}s)"
        )
        mutation_list_str = "\n".join(mutation_list)

        # Save results into dataframe
        df.loc[index, "mutation_list"] = mutation_list_str
        df.loc[index, "answer_list"] = record_str
        df.loc[index, "final_answer_ra"] = final_answer_ra
        df.loc[index, "token_cost_ra"] = tokens_ra
        df.loc[index, "time_cost_ra"] = time_mu + end_ra - start_ra

        df.to_csv(output_path, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)
    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ABLATION (no Metamorphic Relations): naive multi-round mutation pipeline")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--n_mutations", type=int, default=5,
                        help="Number of independent restatement rounds (default: 5)")
    args = parser.parse_args()
    model_key = 'qwen3-32b'
    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"./outputs/{model_key}_mutation_nomr_outputs_{dataset_name}.csv"

    os.makedirs("./outputs", exist_ok=True)
    run_pipeline(dataset_path, output_path, model_key, n_mutations=args.n_mutations)
