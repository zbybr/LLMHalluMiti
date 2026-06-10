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
         Every ordered pair (a_i, a_j) receives an independent LLM call that
         scores a_i in [1, n]. The candidate with the lowest total score wins.

LLM input per call
-------------------
  problem_description  +  starter_code  +  base_response   (repair stages)
  problem_description  +  starter_code  +  candidate A/B   (judge stage)

Required CSV columns
--------------------
  task_id, problem_description, starter_code, base_response
"""

import argparse
import csv
import os
import random
import re
import time
from pathlib import Path
import llm_prompts.prompts as prompts
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv(override=True)
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

def safe_chat_call(messages: list, model_key: str,
                   max_retries: int = 20) -> tuple:
    """Retry-safe chat completion. Returns (content, token_estimate)."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_key,
                messages=messages,
                temperature=0.0,
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
    )
    return extract_code(raw), tokens


def stage3_pairwise_ranking(ctx: str, candidates: list,
                             model_key: str) -> tuple:
    """
    Stage 3 — LLM-as-Judge Pairwise Ranking  (Algorithm 1, lines 8-17).

    For every ordered pair (i, j) one LLM call assigns a score in [1, n]
    to candidate i.  The candidate with the lowest total score is returned.

    Returns (best_code, total_tokens, score_matrix).
    """
    n = len(candidates)
    if n == 1:
        return candidates[0], 0, [[0]]

    R = [[0] * n for _ in range(n)]   # R[i][j] = score of candidate i vs j
    total_tokens = 0
    sys_prompt = prompts.PAIRWISE_JUDGE_LEETCODE_PROMPT.format(n=n)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            user = (
                f"{ctx}\n\n"
                f"## Candidate A\n```python\n{candidates[i]}\n```\n\n"
                f"## Candidate B\n```python\n{candidates[j]}\n```"
            )
            raw, tokens = safe_chat_call(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user",   "content": user}],
                model_key,
            )
            total_tokens += tokens
            m = re.search(r"\b([1-9]\d*)\b", raw.strip())
            score = int(m.group(1)) if m else n
            R[i][j] = max(1, min(n, score))

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
    out_cols = ["final_code", "token_cost", "time_cost",
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

    todo = df[df["final_code"].isna() | (df["final_code"].astype(str).str.strip() == "")]
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
        final_code, tok3, score_matrix = stage3_pairwise_ranking(ctx, candidates, model_key)
        total_tokens += tok3

        elapsed = round(time.time() - t0, 2)
        print(f"  [{task_id}]  tokens={total_tokens}  time={elapsed}s")

        df.loc[idx, "final_answer"]    = final_code
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
    ap.add_argument("--model_key", default="gpt-4o")
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
