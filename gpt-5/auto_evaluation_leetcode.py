"""
auto_evaluation_leetcode.py
============================
Execution-based hallucination evaluation for LeetCode repair experiments.

Evaluates one or more repair output CSVs (e.g. from
repair_with_mutation_leetcode.py and drhall_leetcode.py) against the same
execution oracle and prints a side-by-side comparison.

Metrics  (MutRepair paper Section 3.5, Equations 4-6)
-------------------------------------------------------
  HR   — Hallucination Rate of original base responses
         = |{i : h(B_i)=1}| / N

  RHR  — Recheck Hallucination Rate after repair  (Eq. 4)
         = |{i : h(A_i)=1}| / N

  HRR  — Hallucination Repair Rate  (Eq. 5)
         = |{i : h(B_i)=1 ∧ h(A_i)=0}| / |{i : h(B_i)=1}|

  OCR  — Over-Correction Rate  (Eq. 6)
         = |{i : h(B_i)=0 ∧ h(A_i)=1}| / |{i : h(B_i)=0}|

where h(·) = 1 if the solution fails ≥1 test assertion (i.e. hallucinated).

Hallucination detection oracle
--------------------------------
Each solution is executed in an isolated subprocess:

    <solution code>

    def check(candidate):          ← from the `test` column
        assert candidate(...) == ...

    check(<entry_point>)           ← e.g. check(Solution().canAliceWin)

Consistency guarantee
----------------------
base_pass is computed ONCE per task_id and cached, then shared across all
methods.  This guarantees that HR is identical for every method and that
HRR / OCR are computed against the same base classification.

Expected input schema (per repair CSV)
----------------------------------------
  Required : task_id, final_answer
  Optional : base_response, test, entry_point
              (merged from --dataset_path when missing)

All repair scripts (repair_with_mutation_leetcode.py, drhall_leetcode.py)
write their repaired solution to the `final_answer` column and append it to
the original dataset CSV, so their outputs can be evaluated standalone.

Usage
-----
  # Single method
  python auto_evaluation_leetcode.py \\
      --repair_paths outputs/gpt-4o_mutrepair_xxx.csv \\
      --method_names MutRepair

  # Compare MutRepair vs DrHall ECMR3
  python auto_evaluation_leetcode.py \\
      --repair_paths outputs/gpt-4o_mutrepair_xxx.csv outputs/gpt-4o_drhall_ecmr3_xxx.csv \\
      --method_names MutRepair DrHall-ECMR3 \\
      --dataset_path gpt-4o_leetcode_sampled_responses_sampled.csv \\
      --output_path  outputs/eval_comparison.csv
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


RESULT_COL = "final_answer"     # unified result column across all repair scripts


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
    Execute solution + test harness in an isolated subprocess.
    Returns True iff all assertions pass (i.e. NOT hallucinated).
    """
    if not code or not isinstance(code, str):
        return False
    if not test_str or not entry_point:
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


# ─────────────────────────────────────────────────────────────────────────────
# Metrics  (paper Section 3.5, Equations 4-6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(base_hall: list, final_hall: list) -> dict:
    """
    Compute HR, RHR, HRR, OCR from boolean hallucination flags.

    base_hall[i]  = h(B_i)  True → base response i is hallucinated
    final_hall[i] = h(A_i)  True → repaired response i is hallucinated
    """
    n = len(base_hall)
    assert n == len(final_hall), "Lists must be the same length."
    if n == 0:
        return {}

    n_bh = sum(base_hall)
    n_fh = sum(final_hall)

    orig_bad  = [(b, f) for b, f in zip(base_hall, final_hall) if b]
    orig_good = [(b, f) for b, f in zip(base_hall, final_hall) if not b]

    hr  = 100.0 * n_bh / n
    rhr = 100.0 * n_fh / n                                                 # Eq. 4
    hrr = (100.0 * sum(1 for _, f in orig_bad  if not f) / len(orig_bad)   # Eq. 5
           if orig_bad  else 0.0)
    ocr = (100.0 * sum(1 for _, f in orig_good if     f) / len(orig_good)  # Eq. 6
           if orig_good else 0.0)

    return {
        "n_total":          n,
        "n_base_hall":      n_bh,
        "n_final_hall":     n_fh,
        "n_repaired":       sum(1 for b, f in zip(base_hall, final_hall) if b and not f),
        "n_over_corrected": sum(1 for b, f in zip(base_hall, final_hall) if not b and f),
        "HR":  round(hr,  2),
        "RHR": round(rhr, 2),
        "HRR": round(hrr, 2),
        "OCR": round(ocr, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _non_empty_mask(df: pd.DataFrame, col: str) -> pd.Series:
    """Boolean mask: rows where df[col] holds a non-empty string."""
    return df[col].notna() & (df[col].astype(str).str.strip() != "")


def load_repair_csv(path: str, dataset_path: str | None, label: str) -> pd.DataFrame:
    """
    Load a repair output CSV; merge base_response / test / entry_point from
    the original dataset CSV if any of them are missing.

    The repaired solution must be in the `final_answer` column.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    if "task_id" not in df.columns:
        raise ValueError(
            f"[{label}] {path} must contain a 'task_id' column. "
            f"Found columns: {list(df.columns)}"
        )
    if RESULT_COL not in df.columns:
        raise ValueError(
            f"[{label}] {path} must contain a '{RESULT_COL}' column holding the "
            f"repaired solution. Found columns: {list(df.columns)}"
        )
    if not _non_empty_mask(df, RESULT_COL).any():
        raise ValueError(
            f"[{label}] '{RESULT_COL}' column in {path} is entirely empty — "
            "has the repair script finished running?"
        )

    mergeable = ("base_response", "test", "entry_point")
    missing   = [c for c in mergeable if c not in df.columns]
    if missing:
        if not dataset_path:
            raise ValueError(
                f"[{label}] {path} is missing columns {missing}. "
                "Provide --dataset_path so they can be merged from the original dataset."
            )
        orig = pd.read_csv(dataset_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        absent_in_orig = [c for c in missing if c not in orig.columns]
        if absent_in_orig:
            raise ValueError(
                f"Original dataset {dataset_path} also lacks columns {absent_in_orig}."
            )
        df = df.merge(orig[["task_id"] + missing], on="task_id", how="left")

    # Drop rows whose final_answer is empty (unprocessed by the repair script)
    has_final = _non_empty_mask(df, RESULT_COL)
    n_dropped = (~has_final).sum()
    if n_dropped:
        print(f"[{label}] Skipping {n_dropped} rows without '{RESULT_COL}' "
              f"(not yet processed by the repair script).")
    return df[has_final].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation core
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_method(
    df: pd.DataFrame,
    label: str,
    base_pass_cache: dict,
    timeout: float,
) -> tuple:
    """
    Evaluate one method's repair output (final_answer column).

    base_pass_cache : dict[task_id → bool]
        Shared across methods.  base_pass for each task_id is computed once
        on first encounter and reused thereafter, guaranteeing a consistent
        base classification (identical HR) for every method.

    Returns (per_task_records, metrics) where per_task_records is a list of
    dicts: {task_id, base_pass, final_pass}.
    """
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating [{label}]"):
        task_id     = row["task_id"]
        test_str    = str(row["test"])
        entry_point = str(row["entry_point"])

        # base_pass — compute once per task, then reuse
        if task_id not in base_pass_cache:
            base_code = extract_code(str(row["base_response"]))
            base_pass_cache[task_id] = run_tests(
                base_code, test_str, entry_point, timeout)
        base_pass = base_pass_cache[task_id]

        final_code = extract_code(str(row[RESULT_COL]))
        final_pass = run_tests(final_code, test_str, entry_point, timeout)

        records.append({
            "task_id":    task_id,
            "base_pass":  base_pass,
            "final_pass": final_pass,
        })

    base_hall  = [not r["base_pass"]  for r in records]
    final_hall = [not r["final_pass"] for r in records]
    metrics = compute_metrics(base_hall, final_hall)
    return records, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    repair_paths: list,
    method_names: list,
    output_path: str,
    dataset_path: str | None = None,
    timeout: float = 10.0,
) -> dict:
    """
    Evaluate one or more repair CSVs and print a side-by-side comparison.

    Parameters
    ----------
    repair_paths : list of CSV paths (one per repair method); each must
                   contain task_id and final_answer columns
    method_names : human-readable name per path (same length as repair_paths)
    output_path  : where to save the merged per-task annotation CSV
    dataset_path : original dataset CSV; needed only when a repair CSV lacks
                   base_response / test / entry_point columns
    timeout      : subprocess timeout per test execution (seconds)
    """
    assert len(repair_paths) == len(method_names), \
        "repair_paths and method_names must have the same length."

    base_pass_cache: dict = {}     # task_id → bool, shared across methods
    all_results:  dict = {}        # method  → metrics dict
    all_records:  dict = {}        # method  → per-task records

    for path, name in zip(repair_paths, method_names):
        print(f"\n{'='*55}\n  Loading & evaluating: {name}\n  ({path})\n{'='*55}")
        df = load_repair_csv(path, dataset_path, name)
        records, metrics = evaluate_method(df, name, base_pass_cache, timeout)
        all_records[name] = records
        all_results[name] = metrics

    # ── Build merged per-task annotation ─────────────────────────────────────
    merged = None
    for name, records in all_records.items():
        sub = pd.DataFrame(records).rename(
            columns={"final_pass": f"{name}_final_pass"})
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(
                sub[["task_id", f"{name}_final_pass"]],
                on="task_id", how="outer")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, encoding="utf-8-sig", index=False,
                  quoting=csv.QUOTE_ALL)
    print(f"\nPer-task annotations saved → {output_path}")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = str(Path(output_path).with_suffix(".metrics.json"))
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Metrics saved              → {metrics_path}")

    # ── Print comparison table ────────────────────────────────────────────────
    print_comparison(all_results)
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(results: dict) -> None:
    """Print a formatted side-by-side comparison for all evaluated methods."""
    if not results:
        print("No results to display.")
        return

    methods = list(results.keys())
    col_w   = max(14, *(len(m) for m in methods))

    header  = f"{'Metric':<16}" + "".join(f"  {m:>{col_w}}" for m in methods)

    print()
    print("=" * len(header))
    print("  Hallucination Repair Evaluation")
    print("=" * len(header))

    # Sample counts and base HR per method.
    # HR is identical when methods share the same task set (base_pass cached);
    # it may differ if the task sets differ.
    for m in methods:
        r = results[m]
        print(f"  {m:<{col_w}} : n={r['n_total']},  "
              f"base HR={r['HR']:.1f}% ({r['n_base_hall']}/{r['n_total']})")
    print()
    print(header)
    print("-" * len(header))

    for label, key in [("RHR ↓  (%)", "RHR"),
                       ("HRR ↑  (%)", "HRR"),
                       ("OCR ↓  (%)", "OCR")]:
        values = "".join(f"  {results[m][key]:>{col_w}.1f}" for m in methods)
        print(f"{label:<16}{values}")

    print("-" * len(header))
    for label, key in [("# fixed",     "n_repaired"),
                       ("# broken",    "n_over_corrected"),
                       ("# still bad", "n_final_hall")]:
        values = "".join(f"  {results[m][key]:>{col_w}d}" for m in methods)
        print(f"{label:<16}{values}")
    print("=" * len(header))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Execution-based hallucination evaluation for LeetCode repair "
                    "(supports multi-method comparison; reads 'final_answer' column)")
    ap.add_argument(
        "--repair_paths", required=True, nargs="+",
        help="One or more repair output CSVs "
             "(from repair_with_mutation_leetcode.py / drhall_leetcode.py)",
    )
    ap.add_argument(
        "--method_names", default=None, nargs="+",
        help="Display name per repair CSV (default: derived from filenames)",
    )
    ap.add_argument(
        "--dataset_path", default=None,
        help="Original dataset CSV; needed only when a repair CSV lacks "
             "base_response / test / entry_point columns",
    )
    ap.add_argument(
        "--output_path", default="./outputs/eval_code/evaluation_results.csv",
        help="Where to save the merged per-task annotation CSV",
    )
    ap.add_argument(
        "--timeout", type=float, default=10.0,
        help="Subprocess timeout per test execution in seconds (default: 10)",
    )
    args = ap.parse_args()

    method_names = args.method_names or [
        Path(p).stem for p in args.repair_paths
    ]
    if len(method_names) != len(args.repair_paths):
        raise SystemExit("--method_names must match the number of --repair_paths.")

    run_evaluation(
        repair_paths = args.repair_paths,
        method_names = method_names,
        output_path  = args.output_path,
        dataset_path = args.dataset_path,
        timeout      = args.timeout,
    )
