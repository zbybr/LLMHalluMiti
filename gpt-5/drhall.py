"""
drhall.py
=========
DrHall ECMR3 for natural-language (QA) hallucination repair.

Mirrors the conventions of repair_with_mutation.py:
  - Input CSV columns : Question, Answer, base_response
  - Merge / resume key : "Question"
  - Prompts imported from llm_prompts.prompts
  - Output column      : final_answer  (DrHall produces ONE final answer)

Method: ECMR3 — Multi-Path QMR3  (DrHall, FSE 2025, Section 3.6)
------------------------------------------------------------------
DrHall's core insight: hallucinated answers are *unstable* — asking the same
question through different but semantically-equivalent phrasings makes a
hallucinated answer change, while a correct answer is consistently re-derived.

Pipeline
---------
Stage 1   Paraphrase the question k ways (word-level, structure-level, combined
          synonymous substitution).

Stage 2   Verify-and-REPAIR the base answer once per question path
          (original + k paraphrases). Every path sees the SAME base answer;
          only the question phrasing differs. The unmodified base joins the
          pool as the "keep" candidate → k+2 candidates.

Stage 3   Semantic-consistency majority voting. QA answers have no executable
          oracle, so an LLM clusters candidates by core claim and returns a
          representative of the largest cluster.

Methodological contrast with MutRepair (both are REPAIR methods)
------------------------------------------------------------------
  MutRepair    : mutates the ANSWER  (semantic MRs) → repair → MV / CS / RA
  DrHall ECMR3 : mutates the QUESTION (paraphrase MRs) → neutral
                 verify-and-repair per path → semantic-consistency voting

Identical information access: both see only Question + base_response.

Evaluation is performed manually (no auto_evaluation step for QA).
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


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_paraphrases(text: str, k: int) -> list:
    """Parse up to k numbered paraphrases from the LLM response."""
    pattern = re.compile(r"\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\Z)", re.DOTALL)
    blocks = [m.group(1).strip() for m in pattern.finditer(text)]
    if blocks:
        return blocks[:k]
    parts = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]
    return parts[:k]


def safe_chat_call(messages, model_key, max_retries=20, base_delay=0.0):
    """
    Safe wrapper for OpenAI chat completion with retries and token tracking.
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


# ─────────────────────────────────────────────────────────────────────────────
# Main repair pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_path, output_path, model_key, k=5):
    df = pd.read_csv(input_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    init_cols = [
        "final_answer", "token_cost", "time_cost",
        "paraphrase_list", "answer_list",
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
        df["final_answer"].isna() | (df["final_answer"].astype(str).str.strip() == "")
    )
    df_todo = df[condition]

    print(f"Total questions: {len(df)}")
    print(f"Already processed: {len(df) - len(df_todo)}")
    print(f"Remaining to process: {len(df_todo)}")

    for index, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="DrHall-ECMR3 QA"):
        start = time.time()
        question = row["Question"]
        base_response = row["base_response"]

        # ── Stage 1: paraphrase the question k ways ──────────────────────────
        messages = [
            {"role": "system", "content": prompts.PARAPHRASE_PROMPT_DRHALL_QA.format(k=k)},
            {"role": "user", "content": f"Question: {question}"},
        ]
        paraphrase_raw, tokens = safe_chat_call(messages, model_key)
        paraphrases = extract_paraphrases(paraphrase_raw, k)
        while len(paraphrases) < k:          # pad if the model returned fewer
            paraphrases.append(str(question))

        # ── Stage 2: verify-and-repair the base answer once per question path ─
        # Every path sees the SAME base answer; only the question phrasing
        # differs. The unmodified base joins the pool as the "keep" candidate.
        all_questions = [question] + paraphrases
        record = []
        for q in all_questions:
            qapair = f"Question: {q}\nBase_response: {base_response}"
            messages = [
                {"role": "system", "content": prompts.REPAIR_PROMPT_DRHALL_QA},
                {"role": "user", "content": qapair},
            ]
            answer, _tokens = safe_chat_call(messages, model_key)
            tokens += _tokens
            record.append(answer.strip())
        # Append the unmodified base response as the "keep" candidate
        record.append(str(base_response).strip())
        record_str = "\n".join(record)

        # ── Stage 3: semantic-consistency majority voting ────────────────────
        messages = [
            {"role": "system", "content": prompts.CONSISTENCY_VOTE_PROMPT_DRHALL_QA},
            {"role": "user", "content": f"Question: {question}\nAnswers: {record_str}"},
        ]
        final_answer, _tokens = safe_chat_call(messages, model_key)
        tokens += _tokens

        end = time.time()
        elapsed = end - start

        # Logging
        print("===================================")
        print(f"Question: {question}")
        print(f"Base Response: {base_response}")
        print(f"Final Answer (DrHall ECMR3): {final_answer} "
              f"(tokens={tokens}, time={elapsed:.4f}s)")

        paraphrase_list_str = "\n".join(str(p) for p in [question] + paraphrases)

        # Save results into dataframe
        df.loc[index, "paraphrase_list"] = paraphrase_list_str
        df.loc[index, "answer_list"] = record_str
        df.loc[index, "final_answer"] = final_answer
        df.loc[index, "token_cost"] = tokens
        df.loc[index, "time_cost"] = elapsed

        df.to_csv(output_path, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)

    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DrHall ECMR3 (repair mode) — QA hallucination mitigation pipeline")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of question paraphrases / repair paths (default: 5)")
    args = parser.parse_args()
    model_key = 'gpt-5'
    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"./outputs/{model_key}_drhall_ecmr3_{dataset_name}.csv"

    os.makedirs("./outputs", exist_ok=True)
    run_pipeline(dataset_path, output_path, model_key, k=args.k)
