"""Recompute HRR/RHR/OCR from published QA and LeetCode labels, without APIs.

Formulas are extracted from tools/finalize_stability_metrics.py and
tools/finalize_leetcode_evaluations.py in the source repository. Empty labels
count as incorrect; unknown nonempty values raise an error. This script writes
only summary tables, not experimental answers or labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metrics import metrics_for_file, question_key, read_rows, write_table
from result_files import ROOT, MODELS, qa_files, code_files


def dataset_membership(model: str) -> dict[str, set[str]]:
    full = qa_files(model)["MutRepair"]
    prefix = full.name.split("_dataset", 1)[0]
    groups = {}
    for dataset in ("truthfulqa", "hotpotqa", "freshqa"):
        path = full.with_name(f"{prefix}_{dataset}.csv")
        _, rows = read_rows(path)
        keys = [question_key(row["Question"]) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate dataset questions: {path}")
        groups[dataset] = set(keys)
    if sum(map(len, groups.values())) != len(set.union(*groups.values())):
        raise ValueError(f"overlapping QA datasets: {model}")
    _, full_rows = read_rows(full)
    if set.union(*groups.values()) != {question_key(row["Question"]) for row in full_rows}:
        raise ValueError(f"QA subsets do not partition the full dataset: {model}")
    return groups


def summarize(models=MODELS, task="all", *, overall_only=False):
    results = []
    for model in models:
        if task in {"all", "qa"}:
            groups = {"overall": None}
            if not overall_only:
                groups.update(dataset_membership(model))
            for method, path in qa_files(model).items():
                for dataset, questions in groups.items():
                    results.append(dict(model=model, task="qa", dataset=dataset,
                                        method=method, file=path.relative_to(ROOT).as_posix(),
                                        **metrics_for_file(path, questions=questions)))
        if task in {"all", "code"}:
            for method, path in code_files(model).items():
                results.append(dict(model=model, task="code", dataset="leetcode",
                                    method=method, file=path.relative_to(ROOT).as_posix(),
                                    **metrics_for_file(path)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--task", choices=("qa", "code", "all"), default="all")
    parser.add_argument("--overall-only", action="store_true")
    parser.add_argument("--files", nargs="+", type=Path,
                        help="Instead summarize specific ablation/sensitivity/result CSVs")
    parser.add_argument("--final-column", help="Optional final-label column for --files")
    parser.add_argument("--output", type=Path, default=ROOT / "stat/metrics_summary.csv")
    args = parser.parse_args()
    try:
        if args.final_column and not args.files:
            parser.error("--final-column requires --files")
        if args.files:
            rows = [dict(file=path.name, **metrics_for_file(path, final_column=args.final_column))
                    for path in args.files]
        else:
            rows = summarize(args.models, args.task, overall_only=args.overall_only)
        write_table(args.output, rows, input_paths=args.files or ())
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Wrote {len(rows)} metric rows to {args.output}; empty labels count as incorrect.")


if __name__ == "__main__":
    main()
