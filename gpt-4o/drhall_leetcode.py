"""
drhall_leetcode.py
==========================
DrHall ECMR3 adapted for LeetCode code-generation hallucination repair.

Method: ECMR3 — Multi-Path QMR3  (DrHall, FSE 2025, Section 3.6)
------------------------------------------------------------------
DrHall's core insight: hallucinated answers are *unstable* — if the same
question is asked through different execution paths, a hallucinated response
is more likely to change than a correct one.

ECMR3 operationalises this by reformulating the *problem description* in
k different ways (word-level and structure-level synonymous substitution),
generating one fresh code solution per reformulation, and then selecting
the most frequently consistent answer via multi-path majority voting.

For code, consistency is measured by test execution:
  - Each candidate is executed against the `test` assertions.
  - The final answer is selected from the majority of passing solutions.
  - If no solution passes, a random candidate is returned (tie-break).

This contrasts with MutRepair (repair_with_mutation_leetcode.py), which mutates the
*already-generated code* and uses LLM-as-Judge pairwise ranking for
selection — making the two methods a clean methodological comparison:

  DrHall  ECMR3  : mutate QUESTION  → multi-path generation → majority vote
  MutRepair      : mutate ANSWER    → fault-injected repair  → pairwise ranking

LLM input per generation call
-------------------------------
  problem_description (paraphrased)  +  starter_code

Required CSV columns
---------------------
  task_id, problem_description, starter_code, base_response,
  entry_point, test
"""

import argparse
import csv
import os
import random
import re
import subprocess
import sys
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
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1 — Problem Paraphrase (QMR3 follow-up question construction)
# Three strategies from ECMR3 (paper Section 3.4 + 3.6):
#   Path 1  — word-level synonymous substitution  (structure preserved)
#   Path 2  — structure-level synonymous substitution (words preserved)
#   Path k  — combined word + structure substitution (most diverse)
# We parameterise to k paths so the caller can match the number of
# mutations used by MutRepair for a fair token-cost comparison.


# Stage 2 — Code Generation from a (possibly paraphrased) problem
# The same prompt is used for all k paths and for the base_response
# re-generation path.  LLM input: paraphrased problem + starter_code.


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Strip markdown fences; return clean Python source."""
    if not isinstance(text, str):
        return ""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_paraphrases(text: str, k: int) -> list:
    """
    Parse up to k numbered paraphrases from the LLM response.
    Falls back to splitting on double-newlines if numbered parsing fails.
    """
    # Try numbered blocks: "1. <text>\n\n2. <text>"
    pattern = re.compile(r"\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\Z)", re.DOTALL)
    blocks = [m.group(1).strip() for m in pattern.finditer(text)]
    if blocks:
        return blocks[:k]
    # Fallback: split on blank lines
    parts = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]
    return parts[:k]


def run_tests(code: str, test_str: str, entry_point: str,
              timeout: float = 10.0) -> bool:
    """
    Run code + test harness in an isolated subprocess.
    Returns True iff all assertions pass.
    """
    if not code or not test_str or not entry_point:
        return False
    harness = code + "\n\n" + test_str + "\ncheck(" + entry_point + ")"
    try:
        result = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


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
# Two pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def stage1_paraphrase_problem(problem_desc: str, model_key: str,
                               k: int) -> tuple:
    """
    Stage 1 — Problem Paraphrase (ECMR3 follow-up question construction).

    Generates k diverse reformulations of the problem description.
    Returns (paraphrases, tokens).  len(paraphrases) == k.
    The original problem_desc is NOT included in the returned list;
    it is appended externally as candidate path 0.
    """
    raw, tokens = safe_chat_call(
        [{"role": "system", "content": prompts.PARAPHRASE_PROMPT_DRHALL.format(k=k)},
         {"role": "user",   "content": problem_desc}],
        model_key,
    )
    paraphrases = extract_paraphrases(raw, k)
    while len(paraphrases) < k:            # pad with original if parsing fails
        paraphrases.append(problem_desc)
    return paraphrases[:k], tokens


def stage2_generate_code(problem_desc: str, starter_code: str,
                          model_key: str) -> tuple:
    """
    Stage 2 — Code Generation for one problem path.

    LLM input: problem_description (possibly paraphrased) + starter_code.
    Returns (code, tokens).
    """
    user = (
        f"## Problem Description\n{problem_desc}\n\n"
        f"## Starter Code\n```python\n{starter_code}\n```"
    )
    raw, tokens = safe_chat_call(
        [{"role": "system", "content": prompts.GENERATION_PROMPT_DRHALL},
         {"role": "user",   "content": user}],
        model_key,
    )
    return extract_code(raw), tokens


def majority_vote(candidates: list, test_str: str, entry_point: str,
                  exec_timeout: float) -> tuple:
    """
    Multi-path majority voting  (DrHall Section 3.6).

    Run each candidate against the test suite.  Collect pass/fail for each.
    Selection rule (mirrors DrHall paper):
      - If ≥1 candidate passes: randomly select from the passing candidates.
      - If all fail: randomly select from all candidates (cannot repair).

    Returns (selected_code, pass_flags, n_passing).
    """
    pass_flags = [
        run_tests(c, test_str, entry_point, timeout=exec_timeout)
        for c in candidates
    ]
    passing = [c for c, p in zip(candidates, pass_flags) if p]
    if passing:
        return random.choice(passing), pass_flags, len(passing)
    return random.choice(candidates), pass_flags, 0


# ─────────────────────────────────────────────────────────────────────────────
# Main repair pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_path: str, model_key: str,
                 k: int = 5, exec_timeout: float = 10.0) -> None:
    """
    DrHall ECMR3 repair pipeline for LeetCode.

    For each row:
      1. Generate k paraphrases of problem_description.
      2. For each paraphrase (+ original problem as path 0), call the LLM
         to generate a code solution.  Total: k+1 candidate solutions.
         (The base_response from the CSV is also included as candidate k+2,
          so the pool mirrors the MutRepair setup of n_mutations+1 candidates.)
      3. Majority vote by test execution; return the winning code.
    """
    df = pd.read_csv(input_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    required = {"task_id", "problem_description", "starter_code",
                "base_response", "entry_point", "test"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    out_cols = ["final_answer", "token_cost", "time_cost",
                "paraphrase_list", "candidate_list", "pass_flags"]
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

    for idx, row in tqdm(todo.iterrows(), total=len(todo), desc="DrHall-ECMR3"):

        task_id      = row["task_id"]
        prob_desc    = str(row["problem_description"])
        starter      = str(row["starter_code"])
        entry_point  = str(row["entry_point"])
        test_str     = str(row["test"])
        base_code    = extract_code(str(row["base_response"]))

        total_tokens = 0
        t0 = time.time()

        # Stage 1 — generate k paraphrases of the problem
        paraphrases, tok1 = stage1_paraphrase_problem(prob_desc, model_key, k)
        total_tokens += tok1

        # Stage 2 — generate one code solution per path
        # Paths: [original_problem, paraphrase_1, ..., paraphrase_k]
        # (base_response is appended as the final candidate, so the total
        #  pool size is k+2, matching MutRepair's n_mutations+1 pool)
        all_problems = [prob_desc] + paraphrases   # k+1 problem paths
        candidates = []
        for p_desc in all_problems:
            code, tok2 = stage2_generate_code(p_desc, starter, model_key)
            candidates.append(code)
            total_tokens += tok2

        # Append the original base_response as an additional candidate
        # (DrHall includes the original answer in the voting pool)
        candidates.append(base_code)

        # Stage 3 — majority vote by test execution
        final_answer, pass_flags, n_passing = majority_vote(
            candidates, test_str, entry_point, exec_timeout
        )

        elapsed = round(time.time() - t0, 2)
        print(f"  [{task_id}]  passing={n_passing}/{len(candidates)}"
              f"  tokens={total_tokens}  time={elapsed}s")

        df.loc[idx, "final_answer"]      = final_answer
        df.loc[idx, "token_cost"]      = total_tokens
        df.loc[idx, "time_cost"]       = elapsed
        df.loc[idx, "paraphrase_list"] = "\n\n---\n\n".join(
            f"# Path {i+1}\n{p}" for i, p in enumerate([prob_desc] + paraphrases))
        df.loc[idx, "candidate_list"]  = "\n\n---\n\n".join(
            f"# Candidate {i+1}\n{c}" for i, c in enumerate(candidates))
        df.loc[idx, "pass_flags"]      = str(pass_flags)

        df.to_csv(output_path, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)

    print(f"\nRepair complete. Output → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="DrHall ECMR3 — LeetCode code-generation hallucination repair")
    ap.add_argument("--dataset_path", required=True,
                    help="Input CSV (gpt-4o_leetcode_sampled_responses_sampled.csv)")
    ap.add_argument("--model_key", default="gpt-4o")
    ap.add_argument("--k", type=int, default=5,
                    help="Number of problem paraphrases / execution paths (default: 5, "
                         "matches MutRepair n_mutations for a fair comparison)")
    ap.add_argument("--exec_timeout", type=float, default=10.0,
                    help="Subprocess timeout per test execution in seconds (default: 10)")
    args = ap.parse_args()

    stem   = Path(args.dataset_path).stem.lower()
    output = f"./outputs/{args.model_key}_drhall_ecmr3_{stem}.csv"
    os.makedirs("./outputs", exist_ok=True)

    run_pipeline(
        input_path   = args.dataset_path,
        output_path  = output,
        model_key    = args.model_key,
        k            = args.k,
        exec_timeout = args.exec_timeout,
    )
