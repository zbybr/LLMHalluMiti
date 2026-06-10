"""
auto_evaluation_leetcode.py
============================
Execution-based hallucination evaluation for LeetCode repair experiments.

Given the repaired output CSV produced by repair_leetcode.py, this script
runs every base_response and final_code against the test cases in the
dataset's `test` column and computes four metrics from the MutRepair paper
(Section 3.5, Equations 4-6):

  HR   — Hallucination Rate of the original base responses
         = |{i : h(B_i)=1}| / N

  RHR  — Recheck Hallucination Rate after repair  (Eq. 4)
         = |{i : h(A_i)=1}| / N

  HRR  — Hallucination Repair Rate  (Eq. 5)
         = |{i : h(B_i)=1 ∧ h(A_i)=0}| / |{i : h(B_i)=1}|

  OCR  — Over-Correction Rate  (Eq. 6)
         = |{i : h(B_i)=0 ∧ h(A_i)=1}| / |{i : h(B_i)=0}|

where h(·) = 1 if the solution fails ≥1 test assertion (hallucinated),
             0 if all assertions pass (correct).

Hallucination detection oracle
--------------------------------
Each solution is executed in an isolated subprocess using the `test` column:

    <solution code>

    def check(candidate):          ← from the `test` column
        assert candidate(...) == ...
        ...

    check(<entry_point>)           ← e.g. check(Solution().canAliceWin)

Required input columns
-----------------------
  Repair output CSV  :  task_id, base_response, final_code
  (merged with original dataset for test + entry_point if not present)

Usage
-----
  # Both files available
  python auto_evaluation_leetcode.py \\
      --repair_path  outputs/gpt-4o_mutrepair_leetcode_sampled.csv \\
      --dataset_path gpt-4o_leetcode_sampled_responses_sampled.csv \\
      --output_path  outputs/evaluation_results.csv

  # Repair CSV already contains test / entry_point columns
  python auto_evaluation_leetcode.py \\
      --repair_path  outputs/gpt-4o_mutrepair_leetcode_sampled.csv \\
      --output_path  outputs/evaluation_results.csv
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
# Code extraction
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


# ─────────────────────────────────────────────────────────────────────────────
# Execution oracle
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(code: str, test_str: str, entry_point: str,
              timeout: float = 10.0) -> bool:
    """
    Execute code + test harness in an isolated subprocess.

    Harness layout:
        <solution code>

        def check(candidate):
            assert candidate(...) == ...   ← from the `test` column

        check(<entry_point>)

    Returns True  if all assertions pass (not hallucinated).
    Returns False on any AssertionError, NameError, timeout, or syntax error.
    """
    if not code or not test_str or not entry_point:
        return False
    harness = code + "\n\n" + test_str + "\ncheck(" + entry_point + ")"
    try:
        result = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Metrics  (paper Section 3.5, Equations 4-6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(base_hall: list, final_hall: list) -> dict:
    """
    Compute HR, RHR, HRR, OCR from boolean hallucination flags.

    Parameters
    ----------
    base_hall  : list[bool]  h(B_i)  True = base response is hallucinated
    final_hall : list[bool]  h(A_i)  True = final response is hallucinated

    Returns a dict with all four metrics (as percentages 0-100).
    """
    n = len(base_hall)
    assert n == len(final_hall), "Lists must have equal length."
    assert n > 0, "Empty sample list."

    n_base_hall  = sum(base_hall)
    n_final_hall = sum(final_hall)

    # HR  — base hallucination rate
    hr = 100.0 * n_base_hall / n

    # RHR — Eq. 4
    rhr = 100.0 * n_final_hall / n

    # HRR — Eq. 5  (denominator: originally hallucinated samples)
    originally_bad = [(b, f) for b, f in zip(base_hall, final_hall) if b]
    hrr = (100.0 * sum(1 for _, f in originally_bad if not f) / len(originally_bad)
           if originally_bad else 0.0)

    # OCR — Eq. 6  (denominator: originally correct samples)
    originally_good = [(b, f) for b, f in zip(base_hall, final_hall) if not b]
    ocr = (100.0 * sum(1 for _, f in originally_good if f) / len(originally_good)
           if originally_good else 0.0)

    return {
        "n_total":           n,
        "n_base_hall":       n_base_hall,
        "n_final_hall":      n_final_hall,
        "n_repaired":        sum(1 for b, f in zip(base_hall, final_hall) if b and not f),
        "n_over_corrected":  sum(1 for b, f in zip(base_hall, final_hall) if not b and f),
        "HR":   round(hr,  2),
        "RHR":  round(rhr, 2),
        "HRR":  round(hrr, 2),
        "OCR":  round(ocr, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation routine
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    repair_path: str,
    output_path: str,
    dataset_path: str | None = None,
    timeout: float = 10.0,
) -> dict:
    """
    Evaluate base_response and final_code for each row.

    Parameters
    ----------
    repair_path  : path to the repair output CSV (from repair_leetcode.py)
    output_path  : where to save the annotated results CSV
    dataset_path : optional original dataset CSV; required only when the repair
                   CSV does not already contain `test` and `entry_point` columns
    timeout      : subprocess timeout per test run (seconds)

    Returns the aggregated metrics dict.
    """
    df = pd.read_csv(repair_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    # Merge test + entry_point from original dataset if needed
    need_merge = not ({"test", "entry_point"} <= set(df.columns))
    if need_merge:
        if not dataset_path:
            raise ValueError(
                "The repair CSV does not contain `test` / `entry_point` columns. "
                "Provide --dataset_path to merge them."
            )
        orig = pd.read_csv(dataset_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        merge_cols = ["task_id"] + [c for c in ("test", "entry_point") if c in orig.columns]
        df = df.merge(orig[merge_cols], on="task_id", how="left")

    required = {"task_id", "base_response", "final_code", "test", "entry_point"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Add evaluation columns if absent
    for col in ("base_pass", "final_pass"):
        if col not in df.columns:
            df[col] = pd.NA

    # Determine which rows still need evaluation
    unevaluated = df[
        df["base_pass"].isna() | df["final_pass"].isna()
    ].index.tolist()

    print(f"Total rows     : {len(df)}")
    print(f"Already eval'd : {len(df) - len(unevaluated)}")
    print(f"To evaluate    : {len(unevaluated)}")

    for idx in tqdm(unevaluated, desc="Evaluating"):
        row         = df.loc[idx]
        test_str    = str(row["test"])
        entry_point = str(row["entry_point"])

        base_code  = extract_code(str(row["base_response"]))
        final_code = extract_code(str(row["final_code"]))

        df.loc[idx, "base_pass"]  = run_tests(base_code,  test_str, entry_point, timeout)
        df.loc[idx, "final_pass"] = run_tests(final_code, test_str, entry_point, timeout)

    # Save annotated CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)
    print(f"\nAnnotated results saved → {output_path}")

    # Collect booleans for metric computation
    evaluated = df[df["base_pass"].notna() & df["final_pass"].notna()].copy()
    if evaluated.empty:
        print("No evaluated rows found.")
        return {}

    def to_bool(val) -> bool:
        return str(val).strip().lower() in ("true", "1", "yes")

    base_hall  = [not to_bool(v) for v in evaluated["base_pass"]]
    final_hall = [not to_bool(v) for v in evaluated["final_pass"]]

    metrics = compute_metrics(base_hall, final_hall)
    print_metrics(metrics)

    # Persist metrics alongside the annotated CSV
    metrics_path = Path(output_path).with_suffix(".metrics.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved  → {metrics_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(m: dict) -> None:
    n          = m["n_total"]
    n_base     = m["n_base_hall"]
    n_final    = m["n_final_hall"]
    n_repaired = m["n_repaired"]
    n_ocr      = m["n_over_corrected"]

    print()
    print("=" * 52)
    print("  Evaluation Summary")
    print("=" * 52)
    print(f"  Total samples evaluated     : {n}")
    print()
    print(f"  Base hallucinated (HR)      : {n_base}/{n}  =  {m['HR']:.1f}%")
    print(f"  Base correct                : {n - n_base}/{n}")
    print()
    print("  ── After repair ──────────────────────────────")
    print(f"  RHR  (still hallucinated) ↓ : {n_final}/{n}  =  {m['RHR']:.1f}%")
    print(f"  HRR  (successfully fixed) ↑ : {n_repaired}/{n_base}  =  {m['HRR']:.1f}%")
    print(f"  OCR  (broken by repair)   ↓ : {n_ocr}/{n - n_base}  =  {m['OCR']:.1f}%")
    print("=" * 52)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Execution-based hallucination evaluation for LeetCode repair")
    ap.add_argument(
        "--repair_path", required=True,
        help="Repair output CSV produced by repair_leetcode.py",
    )
    ap.add_argument(
        "--dataset_path", default=None,
        help="Original dataset CSV; only needed if repair CSV lacks "
             "'test' and 'entry_point' columns",
    )
    ap.add_argument(
        "--output_path", default=None,
        help="Where to save the annotated results CSV "
             "(default: <repair_path stem>_evaluated.csv)",
    )
    ap.add_argument(
        "--timeout", type=float, default=10.0,
        help="Subprocess timeout per test execution in seconds (default: 10)",
    )
    args = ap.parse_args()

    output_path = args.output_path or (
        Path(args.repair_path).parent
        / (Path(args.repair_path).stem + "_evaluated.csv")
    )

    run_evaluation(
        repair_path  = args.repair_path,
        output_path  = str(output_path),
        dataset_path = args.dataset_path,
        timeout      = args.timeout,
    )
