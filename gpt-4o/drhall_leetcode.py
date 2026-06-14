"""
drhall_leetcode.py
==================
DrHall ECMR3 adapted for LeetCode code-generation hallucination repair.
ORACLE-FREE version: the dataset's hidden test cases are NEVER used during
repair — they are reserved exclusively for auto_evaluation_leetcode.py.

Method: ECMR3 — Multi-Path QMR3  (DrHall, FSE 2025, Section 3.6)
------------------------------------------------------------------
DrHall's core insight: hallucinated answers are *unstable* — when the same
question is asked through different execution paths, hallucinated responses
diverge while correct ones converge.  The majority/consistent answer wins.

Pipeline
---------
Stage 1   Paraphrase the problem description k ways (word-level,
          structure-level, combined synonymous substitution).

Stage 2   Verify-and-REPAIR the base_response once per question path
          (original + k paraphrases).  Every path sees the SAME base code;
          only the problem phrasing differs.  The unmodified base joins the
          pool as the "keep" candidate → k+2 candidates.

Stage 3a  Generate n probe inputs from the problem description.
          INPUTS ONLY — no expected outputs, no ground truth.

Stage 3b  Behavioral-consistency majority voting: run every candidate on
          the probe inputs, cluster candidates by identical output vectors,
          select a random member of the largest cluster.  Two solutions
          "agree" ⟺ they produce identical outputs on all probe inputs —
          the faithful code analogue of DrHall's answer-consistency voting.

Methodological contrast with MutRepair (both are REPAIR methods)
------------------------------------------------------------------
  MutRepair      : mutates the ANSWER  (code-level MRs) → fault-injected
                   self-repair → LLM-as-judge pairwise ranking
  DrHall ECMR3   : mutates the QUESTION (paraphrase MRs) → neutral
                   verify-and-repair per path → behavioral-consistency voting

Identical information access: both see only problem_description +
starter_code + base_response; neither touches the `test` column.

Required CSV columns
---------------------
  task_id, problem_description, starter_code, base_response, entry_point
  (`test` is NOT required and NOT read.)
"""

import argparse
import csv
import json
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
    """Parse up to k numbered paraphrases from the LLM response."""
    pattern = re.compile(r"\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\Z)", re.DOTALL)
    blocks = [m.group(1).strip() for m in pattern.finditer(text)]
    if blocks:
        return blocks[:k]
    parts = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]
    return parts[:k]


def extract_voting_inputs(text: str, n: int) -> list:
    """
    Parse probe-input lines from the LLM response.
    Each valid line looks like an argument list: `nums = [1,2], target = 3`.
    """
    lines = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip accidental numbering / bullets / fences
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = line.strip("`").strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    # Deduplicate, preserve order
    seen, unique = set(), []
    for l in lines:
        if l not in seen:
            seen.add(l)
            unique.append(l)
    return unique[:n]


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
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def stage1_paraphrase_problem(problem_desc: str, model_key: str,
                               k: int) -> tuple:
    """Stage 1 — generate k paraphrases of the problem description."""
    raw, tokens = safe_chat_call(
        [{"role": "system", "content": prompts.PARAPHRASE_PROMPT_DRHALL.format(k=k)},
         {"role": "user",   "content": problem_desc}],
        model_key,
    )
    paraphrases = extract_paraphrases(raw, k)
    while len(paraphrases) < k:
        paraphrases.append(problem_desc)
    return paraphrases[:k], tokens


def stage2_repair_base(problem_desc: str, starter_code: str,
                        base_code: str, model_key: str) -> tuple:
    """
    Stage 2 — verify-and-repair the base response through ONE question path.

    The same base_code is shown on every path; only the problem phrasing
    differs.  DrHall's insight: a hallucinated base elicits inconsistent
    repairs across phrasings, while a correct base is consistently kept.
    """
    user = (
        f"## Problem Description\n{problem_desc}\n\n"
        f"## Starter Code\n```python\n{starter_code}\n```\n\n"
        f"## Candidate Solution\n```python\n{base_code}\n```"
    )
    raw, tokens = safe_chat_call(
        [{"role": "system", "content": prompts.REPAIR_PROMPT_DRHALL},
         {"role": "user",   "content": user}],
        model_key,
    )
    return extract_code(raw), tokens


def stage3_generate_voting_inputs(problem_desc: str, starter_code: str,
                                    model_key: str, n: int) -> tuple:
    """
    Stage 3a — generate n probe inputs (ORACLE-FREE: inputs only).

    These probe inputs measure behavioral agreement between candidates.
    They contain NO expected outputs and are unrelated to the dataset's
    hidden test cases.
    """
    user = (
        f"## Problem Description\n{problem_desc}\n\n"
        f"## Starter Code\n```python\n{starter_code}\n```"
    )
    raw, tokens = safe_chat_call(
        [{"role": "system", "content": prompts.VOTING_INPUT_PROMPT_DRHALL.format(n=n)},
         {"role": "user",   "content": user}],
        model_key,
    )
    inputs = extract_voting_inputs(raw, n)
    return inputs, tokens


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral-consistency voting  (oracle-free)
# ─────────────────────────────────────────────────────────────────────────────

def behavioral_signature(code: str, entry_point: str,
                          call_args_list: list,
                          timeout: float = 10.0) -> tuple:
    """
    Run `code` on every probe input and return the tuple of repr(output)s.

    Each probe input is an argument string like `nums = [1,2], target = 3`.
    Per-call exceptions are recorded as "ERROR"; a whole-process timeout or
    crash yields all-"TIMEOUT"/"ERROR" signatures.

    NOTE: only LLM-generated probe inputs are used — never the dataset tests.
    """
    if not code or not call_args_list:
        return ("ERROR",) * max(1, len(call_args_list))

    sep = "\x1e"   # record separator, unlikely to appear in outputs
    harness = "\n".join([
        code,
        "",
        f"candidate = {entry_point}",
        f"ARGS_LIST = {json.dumps(call_args_list)}",
        "outs = []",
        "for args in ARGS_LIST:",
        "    try:",
        "        outs.append(repr(eval('candidate(' + args + ')')))",
        "    except BaseException:",
        "        outs.append('ERROR')",
        f"print({sep!r}.join(outs))",
    ])
    try:
        result = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return ("ERROR",) * len(call_args_list)
        parts = result.stdout.rstrip("\n").split(sep)
        if len(parts) != len(call_args_list):
            return ("ERROR",) * len(call_args_list)
        return tuple(parts)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT",) * len(call_args_list)
    except Exception:
        return ("ERROR",) * len(call_args_list)


def majority_vote(candidates: list, entry_point: str,
                  call_args_list: list, exec_timeout: float) -> tuple:
    """
    Multi-path majority voting via behavioral consistency (DrHall Section 3.6).

    DrHall's principle: hallucinated answers are unstable across execution
    paths, so the answer that the most paths AGREE on is selected.  For code,
    "agreement" = identical outputs on a shared set of probe inputs.

    ORACLE-FREE: probe inputs are LLM-generated and have no expected outputs.
    The dataset's hidden test cases are never touched here.

    Selection rule:
      1. Compute each candidate's behavioral signature.
      2. Cluster candidates by identical signature.
      3. Prefer clusters whose signature is not entirely ERROR/TIMEOUT.
      4. Pick a random candidate from the largest such cluster
         (ties broken randomly).  If all clusters are all-error, pick
         randomly from all candidates.

    Returns (selected_code, signatures, cluster_sizes).
    """
    signatures = [
        behavioral_signature(c, entry_point, call_args_list, exec_timeout)
        for c in candidates
    ]

    clusters: dict = {}
    for i, sig in enumerate(signatures):
        clusters.setdefault(sig, []).append(i)

    def is_all_error(sig: tuple) -> bool:
        return all(s in ("ERROR", "TIMEOUT") for s in sig)

    usable = {sig: idxs for sig, idxs in clusters.items() if not is_all_error(sig)}
    pool   = usable if usable else clusters

    max_size     = max(len(idxs) for idxs in pool.values())
    top_clusters = [idxs for idxs in pool.values() if len(idxs) == max_size]
    chosen_cluster = random.choice(top_clusters)
    chosen_idx     = random.choice(chosen_cluster)

    cluster_sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    return candidates[chosen_idx], signatures, cluster_sizes


# ─────────────────────────────────────────────────────────────────────────────
# Main repair pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_path: str, model_key: str,
                 k: int = 5, n_inputs: int = 5,
                 exec_timeout: float = 10.0) -> None:
    """
    DrHall ECMR3 repair pipeline for LeetCode (ORACLE-FREE).

    For each row:
      1. Generate k paraphrases of problem_description.
      2. Generate one solution per path (original + k paraphrases);
         append base_response → pool of k+2 candidates.
      3. Generate n_inputs probe inputs (inputs only, NO expected outputs).
      4. Behavioral-consistency majority vote: run all candidates on the
         probe inputs, cluster by identical outputs, select from the
         largest cluster.  The dataset's `test` column is NEVER used here.
    """
    df = pd.read_csv(input_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    required = {"task_id", "problem_description", "starter_code",
                "base_response", "entry_point"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    out_cols = ["final_answer", "token_cost", "time_cost",
                "paraphrase_list", "candidate_list",
                "voting_inputs", "cluster_sizes"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NA

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
        base_code    = extract_code(str(row["base_response"]))

        total_tokens = 0
        t0 = time.time()

        # Stage 1 — k paraphrases
        paraphrases, tok1 = stage1_paraphrase_problem(prob_desc, model_key, k)
        total_tokens += tok1

        # Stage 2 — verify-and-repair the base response once per question
        # path (original + k paraphrases); the base itself joins the pool
        # as the "keep unchanged" candidate → k+2 candidates total
        all_problems = [prob_desc] + paraphrases
        candidates = []
        for p_desc in all_problems:
            code, tok2 = stage2_repair_base(p_desc, starter, base_code, model_key)
            candidates.append(code)
            total_tokens += tok2
        candidates.append(base_code)

        # Stage 3a — oracle-free probe inputs
        voting_inputs, tok3 = stage3_generate_voting_inputs(
            prob_desc, starter, model_key, n_inputs
        )
        total_tokens += tok3

        # Stage 3b — behavioral-consistency vote (oracle-free)
        final_answer, signatures, cluster_sizes = majority_vote(
            candidates, entry_point, voting_inputs, exec_timeout
        )

        elapsed = round(time.time() - t0, 2)
        print(f"  [{task_id}]  clusters={cluster_sizes}"
              f"  tokens={total_tokens}  time={elapsed}s")

        df.loc[idx, "final_answer"]    = final_answer
        df.loc[idx, "token_cost"]      = total_tokens
        df.loc[idx, "time_cost"]       = elapsed
        df.loc[idx, "paraphrase_list"] = "\n\n---\n\n".join(
            f"# Path {i+1}\n{p}" for i, p in enumerate([prob_desc] + paraphrases))
        df.loc[idx, "candidate_list"]  = "\n\n---\n\n".join(
            f"# Candidate {i+1}\n{c}" for i, c in enumerate(candidates))
        df.loc[idx, "voting_inputs"]   = "\n".join(voting_inputs)
        df.loc[idx, "cluster_sizes"]   = str(cluster_sizes)

        df.to_csv(output_path, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)

    print(f"\nRepair complete. Output → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="DrHall ECMR3 (repair mode, oracle-free) — "
                    "LeetCode hallucination repair")
    ap.add_argument("--dataset_path", required=True,
                    help="Input dataset CSV "
                         "(gpt-4o_leetcode_sampled_responses_sampled.csv)")
    ap.add_argument("--model_key", default="gpt-4o")
    ap.add_argument("--k", type=int, default=5,
                    help="Number of problem paraphrases (default: 5)")
    ap.add_argument("--n_inputs", type=int, default=5,
                    help="Number of probe inputs for consistency voting (default: 5)")
    ap.add_argument("--exec_timeout", type=float, default=10.0,
                    help="Subprocess timeout per candidate run (default: 10)")
    args = ap.parse_args()

    os.makedirs("./outputs", exist_ok=True)
    stem   = Path(args.dataset_path).stem.lower()
    output = f"./outputs/{args.model_key}_drhall_ecmr3_{stem}.csv"

    run_pipeline(
        input_path   = args.dataset_path,
        output_path  = output,
        model_key    = args.model_key,
        k            = args.k,
        n_inputs     = args.n_inputs,
        exec_timeout = args.exec_timeout,
    )
