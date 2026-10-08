"""Recompute four-run stability means and sample SDs from final public labels.

Adapted from the source tools/finalize_stability_metrics.py: retain the count
formulas and statistics.mean/statistics.stdev aggregation, but do not import
private review records, call judges, harmonize labels or rewrite source files.
The original run and three repeated runs are each weighted equally.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from metrics import metrics_for_file, parse_success, read_rows, write_table
from result_files import ROOT, MODELS


def stability_results(models=MODELS):
    results, summaries = [], []
    for model in models:
        directory = ROOT / model / "outputs/eval_stability"
        base_labels = None
        model_rows = []
        for run in ("original", "run_1", "run_2", "run_3"):
            path = directory / f"mutrepair_{run}.csv"
            fields, rows = read_rows(path)
            if not {"id", "base", "mutrepair"} <= set(fields):
                raise ValueError(f"missing stability label columns: {path}")
            ids = {(row["id"] or "").strip() for row in rows}
            if len(rows) != 1826 or ids != {str(i) for i in range(1, 1827)}:
                raise ValueError(f"expected IDs 1..1826 exactly once: {path}")
            current_base = {row["id"].strip(): parse_success(row["base"], "correct") for row in rows}
            if base_labels is not None and current_base != base_labels:
                raise ValueError(f"base labels differ across stability runs: {model}")
            base_labels = current_base
            result = dict(model=model, run=run, file=path.relative_to(ROOT).as_posix(),
                          **metrics_for_file(path))
            results.append(result)
            model_rows.append(result)
        summary = dict(model=model, runs=len(model_rows), empty_label_policy="incorrect")
        for metric in ("RHR", "HRR", "OCR"):
            values = [row[metric] for row in model_rows]
            summary[f"{metric}_mean"] = statistics.mean(values) if None not in values else None
            summary[f"{metric}_std"] = statistics.stdev(values) if None not in values else None
        summaries.append(summary)
    return results, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "stat")
    args = parser.parse_args()
    try:
        results, summaries = stability_results(args.models)
        write_table(args.output_dir / "stability_results.csv", results)
        write_table(args.output_dir / "stability_summary.csv", summaries)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Wrote {len(results)} run metrics and {len(summaries)} four-run summaries. Inputs unchanged.")


if __name__ == "__main__":
    main()
