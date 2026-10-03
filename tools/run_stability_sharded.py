"""Run full stability experiments through resumable dataset shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "gpt-4o": "gpt-4o_dataset20251225_utf8_responses.csv",
    "gpt-5": "gpt-5_dataset20251225_utf8_responses.csv",
    "gemini": "gemini_dataset20251225_utf8_responses.csv",
    "qwen3": "qwen3_32b_dataset20251225_utf8_sig_responses_fixed.csv",
}
PREFIXES = {"gpt-4o": "gpt-4o", "gpt-5": "gpt-5", "gemini": "gemini", "qwen3": "qwen3_32b"}


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    temporary.replace(path)


def create_shards(model: str, count: int) -> tuple[Path, list[Path], int]:
    dataset = ROOT / model / "datasets" / DATASETS[model]
    fields, rows = read(dataset)
    if "__stability_index" not in fields:
        fields.append("__stability_index")
    shard_root = ROOT / model / "outputs" / "stability" / f"shards_{count}"
    size = math.ceil(len(rows) / count)
    paths: list[Path] = []
    for shard in range(count):
        selected = []
        for index, row in enumerate(rows[shard * size : (shard + 1) * size], shard * size):
            item = dict(row)
            item["__stability_index"] = str(index)
            selected.append(item)
        if not selected:
            continue
        path = shard_root / "datasets" / f"{dataset.stem}_part_{shard:02d}.csv"
        write(path, fields, selected)
        paths.append(path)
    return dataset, paths, len(rows)


def run_task(model: str, run: int, shard: Path, workers: int, shard_root: Path) -> dict[str, Any]:
    output = shard_root / "outputs" / f"run_{run}" / f"{PREFIXES[model]}_{shard.stem}_full_run_{run}.csv"
    log_root = shard_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout = log_root / f"run_{run}_{shard.stem}.stdout.log"
    stderr = log_root / f"run_{run}_{shard.stem}.stderr.log"
    command = [
        sys.executable,
        str(ROOT / model / "repair_with_mutation_stability.py"),
        "--dataset_path", str(shard),
        "--output_path", str(output),
        "--runs", str(run),
        "--workers", str(workers),
        "--env", str(ROOT / ".env"),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=out,
            stderr=err,
            creationflags=creationflags,
            check=False,
        )
    return {
        "run": run,
        "shard": shard,
        "output": output,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def merge_outputs(model: str, dataset: Path, shards: list[Path], runs: list[int], shard_root: Path, total: int) -> None:
    output_root = ROOT / model / "outputs" / "stability"
    for run in runs:
        fields: list[str] = []
        combined: list[dict[str, str]] = []
        for shard in shards:
            output = shard_root / "outputs" / f"run_{run}" / f"{PREFIXES[model]}_{shard.stem}_full_run_{run}.csv"
            current_fields, rows = read(output)
            if not fields:
                fields = current_fields
            combined.extend(rows)
        combined.sort(key=lambda row: int(row["__stability_index"]))
        if len(combined) != total or len({row["__stability_index"] for row in combined}) != total:
            raise ValueError(f"{model} run {run}: merged rows are incomplete or duplicated")
        fields = [field for field in fields if field != "__stability_index"]
        for row in combined:
            row.pop("__stability_index", None)
        output = output_root / f"{PREFIXES[model]}_mutation_outputs_{dataset.stem}_full_run_{run}.csv"
        write(output, fields, combined)
        print(f"merged {model} run {run}: {output}", flush=True)


def orchestrate(model: str, shard_count: int, runs: list[int], max_processes: int, workers: int) -> list[dict[str, Any]]:
    dataset, shards, total = create_shards(model, shard_count)
    shard_root = ROOT / model / "outputs" / "stability" / f"shards_{shard_count}"
    tasks = [(run, shard) for run in runs for shard in shards]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_processes) as executor:
        futures = {
            executor.submit(run_task, model, run, shard, workers, shard_root): (run, shard)
            for run, shard in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"run {result['run']} {result['shard'].stem}: exit={result['returncode']}",
                flush=True,
            )
    if all(result["returncode"] == 0 for result in results):
        merge_outputs(model, dataset, shards, runs, shard_root, total)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(DATASETS), required=True)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-processes", type=int, default=30)
    parser.add_argument("--workers-per-process", type=int, default=1)
    parser.add_argument("--auto-fallback-five", action="store_true")
    args = parser.parse_args()
    results = orchestrate(args.model, args.shards, args.runs, args.max_processes, args.workers_per_process)
    failures = [result for result in results if result["returncode"] != 0]
    if not failures:
        return
    if any(result["returncode"] == 75 for result in failures):
        raise SystemExit(75)
    if args.auto_fallback_five and args.shards != 5:
        fallback = orchestrate(args.model, 5, args.runs, min(args.max_processes, 15), args.workers_per_process)
        if all(result["returncode"] == 0 for result in fallback):
            return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
