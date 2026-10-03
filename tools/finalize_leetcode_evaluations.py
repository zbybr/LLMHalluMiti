"""Split full LeetCode execution evaluations into per-method pass/fail CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt-4o", "gpt-5", "gemini", "qwen3")
METHODS = ("CoT", "DrHall", "MutRepair")


def status(value: str, path: Path, task_id: str) -> str:
    normalized = value.strip().lower()
    if normalized == "true":
        return "pass"
    if normalized == "false":
        return "fail"
    raise ValueError(f"{path}: invalid test result for {task_id}: {value!r}")


def load_model(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"task_id", "base_pass"} | {
            f"{method}_final_pass" for method in METHODS
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing columns {required - set(reader.fieldnames or ())}")
        source = list(reader)

    ids = [row["task_id"].strip() for row in source]
    if len(source) != 200 or len(set(ids)) != 200 or any(not task_id for task_id in ids):
        raise ValueError(f"{path}: expected 200 unique nonempty task IDs")

    result: dict[str, list[dict[str, str]]] = {method: [] for method in METHODS}
    for row, task_id in zip(source, ids):
        base = status(row["base_pass"], path, task_id)
        for method in METHODS:
            result[method].append({
                "task_id": task_id,
                "base_pass": base,
                f"{method.lower()}_pass": status(row[f"{method}_final_pass"], path, task_id),
            })
    return result


def write_model(model: str, data: dict[str, list[dict[str, str]]]) -> None:
    destination = ROOT / model / "outputs" / "eval_code"
    destination.mkdir(parents=True, exist_ok=True)
    base_labels = [row["base_pass"] for row in data["CoT"]]
    for method in METHODS:
        name = method.lower()
        rows = data[method]
        if [row["base_pass"] for row in rows] != base_labels:
            raise ValueError(f"{model}/{method}: base labels differ across methods")
        path = destination / f"{name}.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["task_id", "base_pass", f"{name}_pass"]
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

        base_wrong = sum(row["base_pass"] == "fail" for row in rows)
        final_wrong = sum(row[f"{name}_pass"] == "fail" for row in rows)
        repaired = sum(
            row["base_pass"] == "fail" and row[f"{name}_pass"] == "pass"
            for row in rows
        )
        overcorrected = sum(
            row["base_pass"] == "pass" and row[f"{name}_pass"] == "fail"
            for row in rows
        )
        print(
            f"{model}/{name}: n=200 base_wrong={base_wrong} "
            f"repaired={repaired} overcorrected={overcorrected} "
            f"final_wrong={final_wrong} "
            f"HR={base_wrong / 2:.1f} RHR={final_wrong / 2:.1f} "
            f"HRR={100 * repaired / base_wrong:.1f} "
            f"OCR={100 * overcorrected / (200 - base_wrong):.1f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    loaded = {
        model: load_model(args.stage / f"{model}_all.csv") for model in MODELS
    }
    for model in MODELS:
        write_model(model, loaded[model])


if __name__ == "__main__":
    main()
