"""Exact McNemar tests for MutRepair's paired QA and code results.

The default mode understands the repository's existing result schemas and emits
HRR, RHR, and OCR comparisons in one report. Only discordant pairs contribute
to the exact two-sided p-value. No SciPy dependency is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any


SEMANTICS = ("correct", "incorrect")
SUBSETS = ("all", "base-incorrect", "base-correct")


def exact_mcnemar(outcomes_a: list[int], outcomes_b: list[int]) -> dict[str, Any]:
    """Return a two-sided exact McNemar test for paired 0/1 success outcomes."""
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("paired outcome lists must have equal length")
    if not outcomes_a:
        raise ValueError("at least one paired outcome is required")
    if any(value not in (0, 1) for value in outcomes_a + outcomes_b):
        raise ValueError("McNemar outcomes must be binary values 0 or 1")

    both_success = sum(a == 1 and b == 1 for a, b in zip(outcomes_a, outcomes_b))
    a_only = sum(a == 1 and b == 0 for a, b in zip(outcomes_a, outcomes_b))
    b_only = sum(a == 0 and b == 1 for a, b in zip(outcomes_a, outcomes_b))
    both_failure = sum(a == 0 and b == 0 for a, b in zip(outcomes_a, outcomes_b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower_tail = sum(
            math.comb(discordant, index)
            for index in range(min(a_only, b_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * lower_tail)

    return {
        "both_success": both_success,
        "a_success_b_failure": a_only,
        "a_failure_b_success": b_only,
        "both_failure": both_failure,
        "discordant_pairs": discordant,
        "exact_mcnemar_p_two_sided": p_value,
    }


def paired_bootstrap_difference(
    outcomes_a: list[int],
    outcomes_b: list[int],
    *,
    iterations: int = 10_000,
    seed: int = 3622,
) -> list[float]:
    """Percentile 95% CI for success-rate(B) minus success-rate(A)."""
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("paired outcome lists must have equal length")
    if not outcomes_a:
        raise ValueError("at least one paired outcome is required")
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")

    differences = [b - a for a, b in zip(outcomes_a, outcomes_b)]
    try:
        import numpy as np

        values, counts = np.unique(differences, return_counts=True)
        rng = np.random.default_rng(seed)
        sampled_counts = rng.multinomial(
            len(differences), counts / counts.sum(), size=iterations
        )
        samples = np.sort(
            sampled_counts.dot(values) / len(differences)
        ).tolist()
    except ImportError:
        rng = random.Random(seed)
        samples = sorted(
            mean(rng.choice(differences) for _ in differences)
            for _ in range(iterations)
        )
    lower = samples[int(0.025 * (iterations - 1))]
    upper = samples[int(0.975 * (iterations - 1))]
    return [lower, upper]


def parse_success(value: Any, semantics: str) -> int:
    """Convert a binary label to 1=success, respecting column semantics."""
    normalized = str(value).strip().lower()
    positive = {"1", "true", "yes", "y", "correct", "pass", "passed"}
    negative = {"0", "false", "no", "n", "incorrect", "fail", "failed"}
    if normalized in positive:
        indicator = 1
    elif normalized in negative:
        indicator = 0
    else:
        raise ValueError(f"cannot parse binary outcome value: {value!r}")
    if semantics == "correct":
        return indicator
    if semantics == "incorrect":
        return 1 - indicator
    raise ValueError(f"unknown outcome semantics: {semantics}")


def load_rows(path: Path, id_column: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {
                str(key).strip().lstrip("\ufeff"): value
                for key, value in row.items()
                if key is not None
            }
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError(f"result file is empty: {path}")
    if id_column != "__row_number__" and id_column not in rows[0]:
        raise ValueError(f"missing ID column {id_column!r} in {path}")
    indexed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        identity = (
            str(row_number)
            if id_column == "__row_number__"
            else str(row.get(id_column, "")).strip()
        )
        if not identity:
            raise ValueError(f"empty ID in column {id_column!r} in {path}")
        if identity in indexed:
            raise ValueError(f"duplicate ID {identity!r} in {path}")
        indexed[identity] = row
    return indexed


def compare_files(
    file_a: Path,
    file_b: Path,
    *,
    id_column: str,
    outcome_column_a: str,
    outcome_column_b: str,
    semantics_a: str = "correct",
    semantics_b: str = "correct",
    subset: str = "all",
    base_column: str | None = None,
    base_semantics: str = "correct",
    method_a: str | None = None,
    method_b: str | None = None,
    bootstrap_iterations: int = 10_000,
    seed: int = 3622,
    allow_partial_overlap: bool = False,
    skip_invalid_labels: bool = False,
) -> dict[str, Any]:
    rows_a = load_rows(file_a, id_column)
    rows_b = load_rows(file_b, id_column)
    ids_a, ids_b = set(rows_a), set(rows_b)
    if ids_a != ids_b and not allow_partial_overlap:
        raise ValueError(
            "paired files contain different ID sets; use --allow-partial-overlap "
            "only when this exclusion is intentional"
        )
    paired_ids = sorted(ids_a & ids_b)
    if not paired_ids:
        raise ValueError("the two files have no overlapping sample IDs")
    if subset != "all" and not base_column:
        raise ValueError("--base-column is required for a base-correct/incorrect subset")

    outcomes_a: list[int] = []
    outcomes_b: list[int] = []
    excluded_invalid_labels = 0
    for identity in paired_ids:
        row_a, row_b = rows_a[identity], rows_b[identity]
        try:
            if subset != "all":
                base_a = parse_success(row_a.get(base_column, ""), base_semantics)
                base_b = parse_success(row_b.get(base_column, ""), base_semantics)
                if base_a != base_b:
                    raise ValueError(
                        f"base outcome differs between files for ID {identity!r}"
                    )
                if subset == "base-incorrect" and base_a == 1:
                    continue
                if subset == "base-correct" and base_a == 0:
                    continue
            outcome_a = parse_success(
                row_a.get(outcome_column_a, ""), semantics_a
            )
            outcome_b = parse_success(
                row_b.get(outcome_column_b, ""), semantics_b
            )
        except ValueError:
            if not skip_invalid_labels:
                raise
            excluded_invalid_labels += 1
            continue
        outcomes_a.append(outcome_a)
        outcomes_b.append(outcome_b)

    if not outcomes_a:
        raise ValueError(f"no paired samples remain after applying subset {subset!r}")

    name_a = method_a or file_a.stem
    name_b = method_b or file_b.stem
    report = {
        "method_a": name_a,
        "method_b": name_b,
        "file_a": str(file_a.resolve()),
        "file_b": str(file_b.resolve()),
        "id_column": id_column,
        "subset": subset,
        "n_overlapping_ids": len(paired_ids),
        "n_pairs": len(outcomes_a),
        "excluded_invalid_labels": excluded_invalid_labels,
        "unmatched_a": len(ids_a - ids_b),
        "unmatched_b": len(ids_b - ids_a),
        "success_rate_a": mean(outcomes_a),
        "success_rate_b": mean(outcomes_b),
        "success_rate_difference_b_minus_a": mean(outcomes_b) - mean(outcomes_a),
        "paired_bootstrap_95_ci_b_minus_a": paired_bootstrap_difference(
            outcomes_a,
            outcomes_b,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        **exact_mcnemar(outcomes_a, outcomes_b),
    }
    return report


def infer_result_columns(path: Path) -> dict[str, str]:
    """Infer ID/base/final columns and label semantics from an existing CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = {
            str(header).strip().lstrip("\ufeff")
            for header in (reader.fieldnames or [])
        }

    if "is_hallucination" in headers:
        final_candidates = [
            "recheck_hallucination_ra",
            "recheck_hallucination",
            *sorted(
                header
                for header in headers
                if header.startswith("recheck_hallucination")
            ),
        ]
        final_column = next(
            (column for column in final_candidates if column in headers), None
        )
        if final_column is None:
            raise ValueError(f"no rechecked hallucination column found in {path}")
        return {
            "schema": "qa",
            "id_column": "Question",
            "base_column": "is_hallucination",
            "base_semantics": "incorrect",
            "final_column": final_column,
        }

    if "base_pass" in headers:
        final_candidates = sorted(
            header for header in headers if header.endswith("_final_pass")
        )
        if len(final_candidates) != 1:
            raise ValueError(
                f"expected exactly one *_final_pass column in {path}, found "
                f"{final_candidates}"
            )
        return {
            "schema": "code",
            "id_column": "task_id",
            "base_column": "base_pass",
            "base_semantics": "correct",
            "final_column": final_candidates[0],
        }

    raise ValueError(
        f"cannot recognise {path} as QA or code evaluation results; expected "
        "is_hallucination or base_pass"
    )


def compare_paper_metrics(
    file_a: Path,
    file_b: Path,
    *,
    final_column_a: str | None = None,
    final_column_b: str | None = None,
    method_a: str | None = None,
    method_b: str | None = None,
    bootstrap_iterations: int = 10_000,
    seed: int = 3622,
    allow_partial_overlap: bool = False,
    skip_invalid_labels: bool = False,
    id_column_override: str | None = None,
) -> dict[str, Any]:
    """Report paired HRR, RHR, and OCR tests for two experiment files."""
    columns_a = infer_result_columns(file_a)
    columns_b = infer_result_columns(file_b)
    if columns_a["schema"] != columns_b["schema"]:
        raise ValueError(
            f"input schemas differ: {columns_a['schema']} versus "
            f"{columns_b['schema']}"
        )
    if id_column_override:
        columns_a["id_column"] = id_column_override
        columns_b["id_column"] = id_column_override
    if columns_a["id_column"] != columns_b["id_column"]:
        raise ValueError("the inferred ID columns differ between input files")

    if final_column_a:
        columns_a["final_column"] = final_column_a.strip()
    if final_column_b:
        columns_b["final_column"] = final_column_b.strip()

    common = {
        "file_a": file_a,
        "file_b": file_b,
        "id_column": columns_a["id_column"],
        "base_column": columns_a["base_column"],
        "base_semantics": columns_a["base_semantics"],
        "method_a": method_a,
        "method_b": method_b,
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "allow_partial_overlap": allow_partial_overlap,
        "skip_invalid_labels": skip_invalid_labels,
    }

    # HRR is a correctness event among samples whose original response failed.
    success_semantics = (
        "incorrect" if columns_a["schema"] == "qa" else "correct"
    )
    hrr = compare_files(
        outcome_column_a=columns_a["final_column"],
        outcome_column_b=columns_b["final_column"],
        semantics_a=success_semantics,
        semantics_b=success_semantics,
        subset="base-incorrect",
        **common,
    )
    # QA's YES means hallucinated, whereas code's False means failed. Convert
    # both representations into the same 1=incorrect event for RHR and OCR.
    failure_semantics = (
        "correct" if columns_a["schema"] == "qa" else "incorrect"
    )
    rhr = compare_files(
        outcome_column_a=columns_a["final_column"],
        outcome_column_b=columns_b["final_column"],
        semantics_a=failure_semantics,
        semantics_b=failure_semantics,
        subset="all",
        **common,
    )
    ocr = compare_files(
        outcome_column_a=columns_a["final_column"],
        outcome_column_b=columns_b["final_column"],
        semantics_a=failure_semantics,
        semantics_b=failure_semantics,
        subset="base-correct",
        **common,
    )

    metric_explanations = {
        "HRR": "base incorrect -> final correct (higher is better)",
        "RHR": "final incorrect over all samples (lower is better)",
        "OCR": "base correct -> final incorrect (lower is better)",
    }
    metrics = {}
    for metric, raw_report in (("HRR", hrr), ("RHR", rhr), ("OCR", ocr)):
        metrics[metric] = {
            "event": metric_explanations[metric],
            "n_pairs": raw_report["n_pairs"],
            "excluded_invalid_labels": raw_report["excluded_invalid_labels"],
            "rate_a": raw_report["success_rate_a"],
            "rate_b": raw_report["success_rate_b"],
            "rate_difference_b_minus_a": raw_report[
                "success_rate_difference_b_minus_a"
            ],
            "paired_bootstrap_95_ci_b_minus_a": raw_report[
                "paired_bootstrap_95_ci_b_minus_a"
            ],
            "both_event": raw_report["both_success"],
            "a_only_event": raw_report["a_success_b_failure"],
            "b_only_event": raw_report["a_failure_b_success"],
            "neither_event": raw_report["both_failure"],
            "discordant_pairs": raw_report["discordant_pairs"],
            "exact_mcnemar_p_two_sided": raw_report[
                "exact_mcnemar_p_two_sided"
            ],
        }

    return {
        "method_a": method_a or file_a.stem,
        "method_b": method_b or file_b.stem,
        "schema": columns_a["schema"],
        "file_a": str(file_a.resolve()),
        "file_b": str(file_b.resolve()),
        "id_column": columns_a["id_column"],
        "base_column": columns_a["base_column"],
        "final_column_a": columns_a["final_column"],
        "final_column_b": columns_b["final_column"],
        "n_shared_samples": rhr["n_overlapping_ids"],
        "unmatched_a": rhr["unmatched_a"],
        "unmatched_b": rhr["unmatched_b"],
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-a", type=Path, required=True)
    parser.add_argument("--file-b", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("paper", "custom"),
        default="paper",
        help="paper auto-detects QA/code columns and reports HRR, RHR, and OCR",
    )
    parser.add_argument("--id-column", help="required only in custom mode")
    parser.add_argument("--outcome-column-a")
    parser.add_argument("--outcome-column-b")
    parser.add_argument(
        "--final-column-a",
        help="optional override for the auto-detected final QA/code column",
    )
    parser.add_argument(
        "--final-column-b",
        help="optional override for the auto-detected final QA/code column",
    )
    parser.add_argument("--semantics-a", choices=SEMANTICS, default="correct")
    parser.add_argument("--semantics-b", choices=SEMANTICS, default="correct")
    parser.add_argument("--subset", choices=SUBSETS, default="all")
    parser.add_argument("--base-column")
    parser.add_argument("--base-semantics", choices=SEMANTICS, default="correct")
    parser.add_argument("--method-a")
    parser.add_argument("--method-b")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=3622)
    parser.add_argument("--allow-partial-overlap", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        if args.mode == "paper":
            report = compare_paper_metrics(
                args.file_a,
                args.file_b,
                final_column_a=args.final_column_a,
                final_column_b=args.final_column_b,
                method_a=args.method_a,
                method_b=args.method_b,
                bootstrap_iterations=args.bootstrap_iterations,
                seed=args.seed,
                allow_partial_overlap=args.allow_partial_overlap,
            )
        else:
            if not args.id_column:
                raise ValueError("--id-column is required in custom mode")
            if not args.outcome_column_a or not args.outcome_column_b:
                raise ValueError(
                    "--outcome-column-a and --outcome-column-b are required "
                    "in custom mode"
                )
            report = compare_files(
                args.file_a,
                args.file_b,
                id_column=args.id_column,
                outcome_column_a=args.outcome_column_a,
                outcome_column_b=args.outcome_column_b,
                semantics_a=args.semantics_a,
                semantics_b=args.semantics_b,
                subset=args.subset,
                base_column=args.base_column,
                base_semantics=args.base_semantics,
                method_a=args.method_a,
                method_b=args.method_b,
                bootstrap_iterations=args.bootstrap_iterations,
                seed=args.seed,
                allow_partial_overlap=args.allow_partial_overlap,
            )
    except ValueError as error:
        parser.error(str(error))

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
