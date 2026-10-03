"""
repair_leetcode.py
==================
MutRepair — Stage 1-2-3 repair pipeline for LeetCode code generation.
Evaluation is handled separately by auto_evaluation_leetcode.py.

Pipeline  (Algorithm 1, ISSTA 2026)
-------------------------------------
Stage 1  Mutation Construction
         Generate n code-level mutations via four metamorphic relation types:
         meaning-preserving rewrite, structural transformation, semantic polarity
         shift, algorithm/data-structure variant.
         The original base code is appended as the (n+1)-th candidate (line 3).

Stage 2  Fault-Injected Self-Repair
         Each mutation is treated as a potentially faulty response. Explicit
         fault-assumption injection puts the model into active debugging mode.

Stage 3  LLM-as-Judge Pairwise Ranking  (lines 8-17)
         One prompt receives all candidates and returns the complete pairwise
         score matrix. The candidate with the lowest row total wins.

LLM input by stage
------------------
  problem_description  +  starter_code  +  base_response   (repair stages)
  problem_description  +  starter_code  +  all candidates  (one judge call)

Required CSV columns
--------------------
  task_id, problem_description, starter_code, base_response
"""

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from common import prompts
import pandas as pd
from common.config import ROOT_ENV, load_root_env
from openai import OpenAI
from tqdm import tqdm

load_root_env()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Strip markdown fences; return clean Python source."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_mutations(text: str, n: int) -> list:
    """Parse up to n numbered fenced Python blocks from a mutation response."""
    pat = re.compile(
        r"\d+[.)]\s*\n?```(?:python)?[ \t]*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    blocks = [m.group(1).strip() for m in pat.finditer(text)]
    if blocks:
        return blocks[:n]
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        return blocks[:n]
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()][:n]


def num_tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# LLM wrapper
# ─────────────────────────────────────────────────────────────────────────────

def safe_chat_call(messages: list, model_key: str, temperature=0.1,
                   max_retries: int = 20) -> tuple:
    """Retry-safe chat completion. Returns (content, token_estimate)."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_key,
                messages=messages,
                temperature=temperature,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("Empty response")
            tokens = (sum(num_tokens_approx(m.get("content", "")) for m in messages)
                      + num_tokens_approx(content))
            return content, tokens
        except Exception as e:
            wait = random.uniform(0.5, 2.0) * (attempt + 1)
            print(f"  [Retry {attempt+1}/{max_retries}] {e}  (wait {wait:.1f}s)")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} retries.")


# ─────────────────────────────────────────────────────────────────────────────
# Three pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def stage1_generate_mutations(ctx: str, base_code: str,
                               model_key: str, n: int) -> tuple:
    """
    Stage 1 — Mutation Construction.
    Returns (mutations, tokens).  len(mutations) == n+1 (n mutants + base code).
    """
    user = f"{ctx}\n\n## Base Response\n```python\n{base_code}\n```"
    raw, tokens = safe_chat_call(
        [{"role": "system", "content": prompts.MUTATION_LEETCODE_PROMPT.format(n=n)},
         {"role": "user",   "content": user}],
        model_key,
        temperature=0.1,
    )
    mutations = extract_mutations(raw, n)
    while len(mutations) < n:          # pad if the model returned fewer
        mutations.append(base_code)
    mutations.append(base_code)        # Algorithm 1, line 3
    return mutations, tokens


def stage2_fault_repair(ctx: str, mutation: str, model_key: str) -> tuple:
    """
    Stage 2 — Fault-Injected Self-Repair for one mutation.
    Returns (repaired_code, tokens).
    """
    user = f"{ctx}\n\n## Code Under Review\n```python\n{mutation}\n```"
    raw, tokens = safe_chat_call(
        [{"role": "system", "content": prompts.REPAIR_LEETCODE_PROMPT},
         {"role": "user",   "content": user}],
        model_key,
        temperature=0.1,
    )
    return extract_code(raw), tokens


def stage3_pairwise_ranking(ctx: str, candidates: list,
                             model_key: str) -> tuple:
    """
    Stage 3 — LLM-as-Judge Pairwise Ranking  (Algorithm 1, lines 8-17).

    One LLM call evaluates all ordered candidate pairs and returns the complete
    score matrix. The candidate with the lowest total score is returned.

    Returns (best_code, total_tokens, score_matrix).
    """
    n = len(candidates)
    if n == 1:
        return candidates[0], 0, [[0]]

    sys_prompt = prompts.PAIRWISE_JUDGE_LEETCODE_PROMPT.format(n=n)
    candidate_text = "\n\n".join(
        f"## Candidate {i + 1}\n```python\n{candidate}\n```"
        for i, candidate in enumerate(candidates)
    )
    user = f"{ctx}\n\n{candidate_text}"
    raw, total_tokens = safe_chat_call(
        [{"role": "system", "content": sys_prompt},
         {"role": "user",   "content": user}],
        model_key,
        temperature=0.1,
    )

    payload = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", payload, re.DOTALL | re.IGNORECASE)
    if fenced:
        payload = fenced.group(1).strip()
    else:
        start, end = payload.find("{"), payload.rfind("}")
        if start >= 0 and end > start:
            payload = payload[start:end + 1]
    try:
        parsed = json.loads(payload)
        matrix = parsed["score_matrix"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("ranking prompt returned no valid score_matrix JSON") from error

    if not isinstance(matrix, list) or len(matrix) != n:
        raise ValueError(f"score_matrix must have exactly {n} rows")
    R = []
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError(f"score_matrix row {i + 1} must contain exactly {n} values")
        normalized_row = []
        for j, value in enumerate(row):
            try:
                score = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"score_matrix[{i}][{j}] is not an integer"
                ) from error
            if i == j:
                normalized_row.append(0)
            elif 1 <= score <= n:
                normalized_row.append(score)
            else:
                raise ValueError(
                    f"score_matrix[{i}][{j}] must be between 1 and {n}"
                )
        R.append(normalized_row)

    totals  = [sum(R[i]) for i in range(n)]
    best_idx = totals.index(min(totals))
    return candidates[best_idx], total_tokens, R


# ─────────────────────────────────────────────────────────────────────────────
# Main repair pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_path: str,
                 model_key: str, n_mutations: int = 5) -> None:
    df = pd.read_csv(input_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    required = {"task_id", "problem_description", "starter_code", "base_response"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # New output columns
    out_cols = ["final_answer", "token_cost", "time_cost",
                "mutation_list", "candidate_list", "score_matrix"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Resume from partial output
    if os.path.exists(output_path):
        print(f"Resuming from: {output_path}")
        saved = pd.read_csv(output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        keep  = ["task_id"] + [c for c in out_cols if c in saved.columns]
        df    = df.merge(saved[keep], on="task_id", how="left", suffixes=("", "_s"))
        for c in out_cols:
            s = c + "_s"
            if s in df.columns:
                mask = df[c].isna() | (df[c].astype(str).str.strip() == "")
                df.loc[mask, c] = df.loc[mask, s]
                df.drop(columns=[s], inplace=True)

    todo = df[df["final_answer"].isna() | (df["final_answer"].astype(str).str.strip() == "")]
    print(f"Total: {len(df)}  |  Done: {len(df)-len(todo)}  |  Remaining: {len(todo)}")

    for idx, row in tqdm(todo.iterrows(), total=len(todo), desc="MutRepair"):

        task_id   = row["task_id"]
        base_code = extract_code(str(row["base_response"]))

        # Context sent to every LLM call: problem + starter code only
        ctx = (
            f"## Problem Description\n{row['problem_description']}\n\n"
            f"## Starter Code\n```python\n{row['starter_code']}\n```"
        )

        total_tokens = 0
        t0 = time.time()

        # Stage 1
        mutations, tok1 = stage1_generate_mutations(ctx, base_code, model_key, n_mutations)
        total_tokens += tok1

        # Stage 2
        candidates = []
        for mut in mutations:
            repaired, tok2 = stage2_fault_repair(ctx, mut, model_key)
            candidates.append(repaired)
            total_tokens += tok2

        # Stage 3
        final_answer, tok3, score_matrix = stage3_pairwise_ranking(ctx, candidates, model_key)
        total_tokens += tok3

        elapsed = round(time.time() - t0, 2)
        print(f"  [{task_id}]  tokens={total_tokens}  time={elapsed}s")

        df.loc[idx, "final_answer"]    = final_answer
        df.loc[idx, "token_cost"]    = total_tokens
        df.loc[idx, "time_cost"]     = elapsed
        df.loc[idx, "mutation_list"] = "\n\n---\n\n".join(
            f"# Mutation {i+1}\n{m}" for i, m in enumerate(mutations))
        df.loc[idx, "candidate_list"] = "\n\n---\n\n".join(
            f"# Candidate {i+1}\n{c}" for i, c in enumerate(candidates))
        df.loc[idx, "score_matrix"]  = str(score_matrix)

        df.to_csv(output_path, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)

    print(f"\nRepair complete. Output → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MutRepair — LeetCode code-generation hallucination repair")
    ap.add_argument("--dataset_path", required=True,
                    help="Input CSV (e.g. gpt-4o_leetcode_sampled_responses_sampled.csv)")
    ap.add_argument("--model_key", default=os.getenv("LLM_MODEL_KEY", "gpt-4o"))
    ap.add_argument("--n_mutations", type=int, default=5,
                    help="Mutations per sample (default: 5)")
    args = ap.parse_args()

    stem   = Path(args.dataset_path).stem.lower()
    output = f"./outputs/{args.model_key}_mutrepair_{stem}.csv"
    os.makedirs("./outputs", exist_ok=True)

    run_pipeline(
        input_path  = args.dataset_path,
        output_path = output,
        model_key   = args.model_key,
        n_mutations = args.n_mutations,
    )
