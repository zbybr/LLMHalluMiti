"""Run MutRepair-vs-baseline McNemar tests for all four QA models."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from mcnemar_test import compare_paper_metrics


ROOT = Path(__file__).resolve().parents[1]
STAT_DIR = Path(os.environ.get("MCNEMAR_OUTPUT_DIR", ROOT / "stat"))
BASELINES = ("CoT", "CoVe", "DrHall")
METRICS = ("HRR", "RHR", "OCR")

MODEL_FILES = {
    "gpt-4o": {
        "MutRepair": ROOT / "gpt-4o/outputs/gpt-4o_mutation_outputs_gpt-4o_dataset20251225_utf8_responses.csv",
        "CoT": ROOT / "gpt-4o/outputs/cot/gpt-4o_cot_outputs_gpt-4o_dataset20251225_utf8_responses.csv",
        "CoVe": ROOT / "gpt-4o/outputs/cove/gpt-4o_cove_outputs_gpt-4o_dataset20251225_utf8_responses.csv",
        "DrHall": ROOT / "gpt-4o/outputs/drhall/gpt-4o_drhall_ecmr3_gpt-4o_dataset20251225_utf8_responses_check.csv",
    },
    "gpt-5": {
        "MutRepair": ROOT / "gpt-5/outputs/gpt-5_mutation_outputs_gpt-5_dataset20251225_utf8_responses.csv",
        "CoT": ROOT / "gpt-5/outputs/cot/gpt-5_cot_outputs_gpt-5_dataset20251225_utf8_responses.csv",
        "CoVe": ROOT / "gpt-5/outputs/cove/gpt-5_cove_outputs_gpt-5_dataset20251225_utf8_responses.csv",
        "DrHall": ROOT / "gpt-5/outputs/drhall/gpt-5_drhall_ecmr3_gpt-5_dataset20251225_utf8_responses_check.csv",
    },
    "gemini": {
        "MutRepair": ROOT / "gemini/outputs/gemini_mutation_outputs_gemini_dataset20251225_utf8_responses.csv",
        "CoT": ROOT / "gemini/outputs/cot/gemini_cot_outputs_gemini_dataset20251225_utf8_responses.csv",
        "CoVe": ROOT / "gemini/outputs/cove/gemini_cove_outputs_gemini_dataset20251225_utf8_responses.csv",
        "DrHall": ROOT / "gemini/outputs/drhall/gemini_drhall_ecmr3_gemini_dataset20251225_utf8_responses_check.csv",
    },
    "qwen3": {
        "MutRepair": ROOT / "qwen3/outputs/qwen3_32b_mutation_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed_newlines.csv",
        "CoT": ROOT / "ollama_outputs/cot/qwen3_32b_cot_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed.csv",
        "CoVe": ROOT / "ollama_outputs/cove/qwen3_32b_cove_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed.csv",
        "DrHall": ROOT / "qwen3/outputs/drhall/qwen3-32b_drhall_ecmr3_qwen3_32b_dataset20251225_utf8_sig_responses_fixed_check.csv",
    },
}

FIELDNAMES = (
    "model",
    "baseline",
    "metric",
    "pairing_key",
    "status",
    "n_pairs",
    "excluded_invalid_labels",
    "baseline_rate_pct",
    "mutrepair_rate_pct",
    "difference_mutrepair_minus_baseline_pp",
    "ci_95_lower_pp",
    "ci_95_upper_pp",
    "both_event",
    "baseline_only_event",
    "mutrepair_only_event",
    "neither_event",
    "discordant_pairs",
    "exact_two_sided_p",
    "within_model_test_family_size",
    "holm_adjusted_p_within_model",
    "significant_within_model_0_05",
    "global_test_family_size",
    "holm_adjusted_p_global",
    "significant_global_0_05",
)


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


def make_rows(bootstrap_iterations: int = 10_000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, files in MODEL_FILES.items():
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing result files for {model}: {missing}")
        model_rows: list[dict[str, Any]] = []
        for baseline in BASELINES:
            try:
                report = compare_paper_metrics(
                    files[baseline],
                    files["MutRepair"],
                    method_a=baseline,
                    method_b="MutRepair",
                    bootstrap_iterations=bootstrap_iterations,
                    seed=3622,
                    skip_invalid_labels=True,
                    id_column_override=(
                        "__row_number__" if model == "qwen3" else None
                    ),
                )
            except (ValueError, KeyError) as error:
                if "no paired samples remain" not in str(error):
                    raise ValueError(f"{model}/{baseline}: {error}") from error
                for metric in METRICS:
                    model_rows.append(
                        {
                            "model": model,
                            "baseline": baseline,
                            "metric": metric,
                            "pairing_key": (
                                "row_index" if model == "qwen3" else "Question"
                            ),
                            "status": "unavailable: all final labels are missing",
                        }
                    )
                continue
            if report["n_shared_samples"] != 1826:
                raise ValueError(
                    f"{model}/{baseline}: expected 1826 paired samples, "
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
                        "pairing_key": (
                            "row_index" if model == "qwen3" else "Question"
                        ),
                        "status": "ok",
                        "n_pairs": result["n_pairs"],
                        "excluded_invalid_labels": result[
                            "excluded_invalid_labels"
                        ],
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = make_rows()
    STAT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(STAT_DIR / "mcnemar_all_models.csv", rows)
    for model in MODEL_FILES:
        write_csv(
            STAT_DIR / f"{model}_mcnemar_results.csv",
            [row for row in rows if row["model"] == model],
        )
    print(
        f"Wrote {len(rows)} rows across 5 CSV files; "
        f"global significant tests: "
        f"{sum(bool(row.get('significant_global_0_05')) for row in rows)}/"
        f"{len([row for row in rows if row['status'] == 'ok'])} available"
    )


if __name__ == "__main__":
    main()
