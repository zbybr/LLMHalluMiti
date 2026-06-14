"""
diagnose_mutrepair.py
======================
Failure-mode decomposition for MutRepair results on LeetCode.

Decomposes the observed HRR into:

  Ceiling HRR    — fraction of base-hallucinated tasks where AT LEAST ONE
                   repaired candidate passes the tests.  This is the best
                   HRR any selection strategy could achieve given the
                   candidate pool (upper bound for Stage 3).

  Actual HRR     — fraction where the SELECTED final_answer passes.

  Selection acc. — among base-hallucinated tasks where ≥1 candidate passes,
                   how often the ranking stage picked a passing one.

  Gap analysis   — Actual HRR = Ceiling HRR × Selection accuracy
                   → tells you whether to fix Stage 1/2 (raise the ceiling)
                     or Stage 3 (improve selection).

NOTE: tests are used here for POST-HOC ANALYSIS only — this script does not
feed anything back into repair, so there is no oracle leakage.

Usage
-----
  python diagnose_mutrepair.py \\
      --repair_path outputs/gpt-4o_mutrepair_xxx.csv \\
      [--dataset_path original.csv]          # if test/entry_point missing
      [--timeout 10] [--timeout_check 30]    # timeout sensitivity probe
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (same semantics as auto_evaluation_leetcode.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_tests(code: str, test_str: str, entry_point: str,
              timeout: float = 10.0) -> bool:
    if not code or not test_str or not entry_point:
        return False
    harness = code + "\n\n" + test_str + "\ncheck(" + entry_point + ")"
    try:
        r = subprocess.run([sys.executable, "-c", harness],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


CAND_PATTERN = re.compile(
    r"# Candidate \d+\n(.*?)(?=\n\n---\n\n# Candidate \d+\n|\Z)", re.DOTALL)


def parse_candidates(blob: str) -> list:
    if not isinstance(blob, str):
        return []
    return [m.group(1).strip() for m in CAND_PATTERN.finditer(blob)]


# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(repair_path: str, dataset_path: str | None,
             timeout: float, timeout_check: float | None) -> None:

    df = pd.read_csv(repair_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    # Merge test / entry_point if missing
    need = [c for c in ("test", "entry_point", "base_response")
            if c not in df.columns]
    if need:
        if not dataset_path:
            raise ValueError(f"Missing {need}; provide --dataset_path.")
        orig = pd.read_csv(dataset_path, encoding="utf-8-sig",
                           quoting=csv.QUOTE_ALL)
        df = df.merge(orig[["task_id"] + need], on="task_id", how="left")

    for col in ("task_id", "final_answer", "candidate_list"):
        if col not in df.columns:
            raise ValueError(f"Repair CSV missing column: {col}")

    has_final = df["final_answer"].notna() & \
                (df["final_answer"].astype(str).str.strip() != "")
    df = df[has_final].reset_index(drop=True)
    print(f"Analysing {len(df)} processed tasks …\n")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Executing"):
        test_str    = str(row["test"])
        entry_point = str(row["entry_point"])

        base_code  = extract_code(str(row["base_response"]))
        final_code = extract_code(str(row["final_answer"]))
        candidates = [extract_code(c) for c in parse_candidates(str(row["candidate_list"]))]

        base_pass  = run_tests(base_code,  test_str, entry_point, timeout)
        final_pass = run_tests(final_code, test_str, entry_point, timeout)
        cand_pass  = [run_tests(c, test_str, entry_point, timeout)
                      for c in candidates]

        rec = {
            "task_id":         row["task_id"],
            "difficulty":      row.get("difficulty", ""),
            "base_pass":       base_pass,
            "final_pass":      final_pass,
            "n_candidates":    len(candidates),
            "n_cand_passing":  sum(cand_pass),
        }

        # Timeout-sensitivity probe: did a longer budget change the verdict?
        if timeout_check and not final_pass:
            rec["final_pass_long_timeout"] = run_tests(
                final_code, test_str, entry_point, timeout_check)
        rows.append(rec)

    res = pd.DataFrame(rows)
    out_path = Path(repair_path).with_suffix(".diagnosis.csv")
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nPer-task diagnosis saved → {out_path}\n")

    # ── Aggregate decomposition ───────────────────────────────────────────────
    bad = res[~res["base_pass"]]                # base-hallucinated tasks
    n_bad = len(bad)
    if n_bad == 0:
        print("No base-hallucinated tasks found — nothing to repair.")
        return

    ceiling_ok   = bad[bad["n_cand_passing"] > 0]
    actual_ok    = bad[bad["final_pass"]]
    n_ceiling    = len(ceiling_ok)
    n_actual     = len(actual_ok)

    ceiling_hrr  = 100.0 * n_ceiling / n_bad
    actual_hrr   = 100.0 * n_actual  / n_bad
    sel_acc      = (100.0 * len(ceiling_ok[ceiling_ok["final_pass"]]) / n_ceiling
                    if n_ceiling else 0.0)

    print("=" * 60)
    print("  MutRepair Failure Decomposition")
    print("=" * 60)
    print(f"  Base-hallucinated tasks          : {n_bad}")
    print(f"  Ceiling HRR (≥1 candidate passes): {ceiling_hrr:5.1f}%  ({n_ceiling}/{n_bad})")
    print(f"  Actual  HRR (final passes)       : {actual_hrr:5.1f}%  ({n_actual}/{n_bad})")
    print(f"  Selection accuracy | pool has fix: {sel_acc:5.1f}%")
    print("-" * 60)

    gen_loss = 100.0 - ceiling_hrr
    sel_loss = ceiling_hrr - actual_hrr
    print(f"  Loss from Stage 1+2 (no fix in pool)   : {gen_loss:5.1f} pp")
    print(f"  Loss from Stage 3   (bad selection)    : {sel_loss:5.1f} pp")
    print("-" * 60)

    if gen_loss >= sel_loss:
        print("  → PRIMARY bottleneck: GENERATION/REPAIR (Stage 1+2).")
        print("    The pool rarely contains a correct fix at all.")
    else:
        print("  → PRIMARY bottleneck: SELECTION (Stage 3 pairwise ranking).")
        print("    Correct fixes exist but the judge fails to pick them.")

    # ── Difficulty breakdown ──────────────────────────────────────────────────
    if "difficulty" in res.columns and res["difficulty"].astype(str).str.strip().any():
        print("\n  HRR by difficulty:")
        for diff, grp in bad.groupby("difficulty"):
            if not str(diff).strip():
                continue
            n = len(grp)
            fixed   = grp["final_pass"].sum()
            ceiling = (grp["n_cand_passing"] > 0).sum()
            print(f"    {str(diff):<8}: actual {100.0*fixed/n:5.1f}%  "
                  f"ceiling {100.0*ceiling/n:5.1f}%   (n={n})")

    # ── Timeout sensitivity ───────────────────────────────────────────────────
    if timeout_check and "final_pass_long_timeout" in res.columns:
        flipped = res["final_pass_long_timeout"].fillna(False).sum()
        if flipped:
            print(f"\n  ⚠ Timeout sensitivity: {int(flipped)} repaired solutions "
                  f"fail at {timeout}s but PASS at {timeout_check}s.")
            print("    Consider re-running evaluation with a larger --timeout.")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MutRepair failure-mode diagnosis")
    ap.add_argument("--repair_path", required=True)
    ap.add_argument("--dataset_path", default=None)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--timeout_check", type=float, default=30.0,
                    help="Second, longer timeout to probe for slow-but-correct "
                         "solutions misclassified as failures (default: 30; "
                         "set 0 to disable)")
    args = ap.parse_args()
    diagnose(args.repair_path, args.dataset_path,
             args.timeout, args.timeout_check or None)
