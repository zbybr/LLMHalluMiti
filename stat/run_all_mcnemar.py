"""Run paired McNemar/Holm tests for the four models' QA or code results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from mcnemar_test import compare_paper_metrics, load_rows
from metrics import question_key as normalized_question, write_table
from result_files import ROOT, MODELS, qa_files, code_files


STAT_DIR = Path(os.environ.get("MCNEMAR_OUTPUT_DIR", ROOT / "stat"))
BASELINES = ("CoT", "CoVe", "DrHall")
METRICS = ("HRR", "RHR", "OCR")


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in original order."""
    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def validate_row_pairing(file_a: Path, file_b: Path) -> None:
    rows_a = load_rows(file_a, "__row_number__")
    rows_b = load_rows(file_b, "__row_number__")
    if set(rows_a) != set(rows_b):
        raise ValueError("row-paired files have different lengths")
    questions_a = [normalized_question(row["Question"]) for row in rows_a.values()]
    questions_b = [normalized_question(row["Question"]) for row in rows_b.values()]
    if len(set(questions_a)) != len(questions_a) or questions_a != questions_b:
        raise ValueError("row pairing failed question identity/order validation")


def make_rows(
    bootstrap_iterations: int = 10_000, *, models=None, task="qa", baselines=None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in (MODELS if models is None else models):
        files = qa_files(model) if task == "qa" else code_files(model)
        selected_baselines = baselines or (BASELINES if task == "qa" else ("CoT", "DrHall"))
        missing = [str(files[method]) for method in ("MutRepair", *selected_baselines) if not files[method].is_file()]
        if missing:
            raise FileNotFoundError(f"missing result files for {model}: {missing}")
        model_rows: list[dict[str, Any]] = []
        for baseline in selected_baselines:
            try:
                if model == "qwen3" and task == "qa":
                    validate_row_pairing(files[baseline], files["MutRepair"])
                report = compare_paper_metrics(
                    files[baseline],
                    files["MutRepair"],
                    method_a=baseline,
                    method_b="MutRepair",
                    bootstrap_iterations=bootstrap_iterations,
                    seed=3622,
                    id_column_override=(
                        "__row_number__" if model == "qwen3" and task == "qa" else None
                    ),
                )
            except (ValueError, KeyError) as error:
                raise ValueError(f"{model}/{baseline}: {error}") from error
            expected = 1826 if task == "qa" else 200
            if report["n_shared_samples"] != expected:
                raise ValueError(
                    f"{model}/{baseline}: expected {expected} paired samples, "
                    f"found {report['n_shared_samples']}"
                )
            if report["unmatched_a"] or report["unmatched_b"]:
                raise ValueError(f"{model}/{baseline}: unmatched sample IDs found")
            for metric in METRICS:
                result = report["metrics"][metric]
                ci_lower, ci_upper = result["paired_bootstrap_95_ci_b_minus_a"]
                model_rows.append(
                    {
                        "model": model,
                        "baseline": baseline,
                        "metric": metric,
                        "task": task,
                        "empty_label_policy": "incorrect",
                        "pairing_key": (
                            "row_index_verified_question" if model == "qwen3" and task == "qa" else report["id_column"]
                        ),
                        "status": "ok",
                        "n_pairs": result["n_pairs"],
                        "excluded_invalid_labels": result[
                            "excluded_invalid_labels"
                        ],
                        "empty_baseline_labels": result["empty_labels_a"],
                        "empty_mutrepair_labels": result["empty_labels_b"],
                        "baseline_rate_pct": result["rate_a"] * 100,
                        "mutrepair_rate_pct": result["rate_b"] * 100,
                        "difference_mutrepair_minus_baseline_pp": (
                            result["rate_difference_b_minus_a"] * 100
                        ),
                        "ci_95_lower_pp": ci_lower * 100,
                        "ci_95_upper_pp": ci_upper * 100,
                        "both_event": result["both_event"],
                        "baseline_only_event": result["a_only_event"],
                        "mutrepair_only_event": result["b_only_event"],
                        "neither_event": result["neither_event"],
                        "discordant_pairs": result["discordant_pairs"],
                        "exact_two_sided_p": result["exact_mcnemar_p_two_sided"],
                    }
                )

        available_model_rows = [
            row for row in model_rows if row["status"] == "ok"
        ]
        within_model = holm_adjust(
            [float(row["exact_two_sided_p"]) for row in available_model_rows]
        )
        for row, adjusted_p in zip(available_model_rows, within_model):
            row["within_model_test_family_size"] = len(available_model_rows)
            row["holm_adjusted_p_within_model"] = adjusted_p
            row["significant_within_model_0_05"] = adjusted_p < 0.05
        rows.extend(model_rows)

    available_rows = [row for row in rows if row["status"] == "ok"]
    global_adjusted = holm_adjust(
        [float(row["exact_two_sided_p"]) for row in available_rows]
    )
    for row, adjusted_p in zip(available_rows, global_adjusted):
        row["global_test_family_size"] = len(available_rows)
        row["holm_adjusted_p_global"] = adjusted_p
        row["significant_global_0_05"] = adjusted_p < 0.05
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--task", choices=("qa", "code"), default="qa")
    parser.add_argument("--baselines", nargs="+", choices=("CoT", "CoVe", "CoVe-SE", "DrHall"))
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=STAT_DIR)
    args = parser.parse_args()
    if args.task == "code" and args.baselines and any(b not in {"CoT", "DrHall"} for b in args.baselines):
        parser.error("code results contain only CoT and DrHall baselines")
    if len(args.models) != len(set(args.models)) or (args.baselines and len(args.baselines) != len(set(args.baselines))):
        parser.error("models and baselines must not repeat")
    try:
        rows = make_rows(args.bootstrap_iterations, models=args.models, task=args.task, baselines=args.baselines)
        suffix = "" if args.task == "qa" else "_code"
        write_table(args.output_dir / f"mcnemar_all_models{suffix}.csv", rows)
        for model in args.models:
            write_table(args.output_dir / f"{model}_mcnemar_results{suffix}.csv",
                        [row for row in rows if row["model"] == model])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {len(rows)} tests. Empty labels count as incorrect; no input labels were changed.")


if __name__ == "__main__":
    main()
