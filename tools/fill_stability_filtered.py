"""Fill content-filtered stability rows from the corresponding RQ1 full run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "gpt-4o": {
        "source": ROOT / "gpt-4o/outputs/gpt-4o_mutation_outputs_gpt-4o_dataset20251225_utf8_responses.csv",
        "directory": ROOT / "gpt-4o/outputs/stability",
    },
    "gpt-5": {
        "source": ROOT / "gpt-5/outputs/gpt-5_mutation_outputs_gpt-5_dataset20251225_utf8_responses.csv",
        "directory": ROOT / "gpt-5/outputs/stability",
    },
    "gemini": {
        "source": ROOT / "gemini/outputs/gemini_mutation_outputs_gemini_dataset20251225_utf8_responses.csv",
        "directory": ROOT / "gemini/outputs/stability",
    },
    "qwen3": {
        "source": ROOT / "qwen3/outputs/qwen3_32b_mutation_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed_newlines.csv",
        "directory": ROOT / "qwen3/outputs/stability",
    },
}
RESULT_FIELDS = ["final_answer_ra", "token_cost_ra", "time_cost_ra", "mutation_list", "answer_list"]


def key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", " ".join(value.split())).casefold()
    return "".join(character for character in normalized if not unicodedata.combining(character))


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".fill-tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    temporary.replace(path)


def fill_model(model: str) -> None:
    config = CONFIGS[model]
    _, source_rows = read(config["source"])
    source = {key(row["Question"]): row for row in source_rows}
    paths = sorted(config["directory"].glob("*_full_run_*.csv"))
    if len(paths) != 3:
        raise ValueError(f"{model}: expected three full-run CSVs, found {len(paths)}")
    for path in paths:
        fields, rows = read(path)
        filled = 0
        for row in rows:
            if row.get("final_answer_ra", "").strip() and row.get("stability_error", "") != "CONTENT_FILTERED":
                continue
            original = source.get(key(row["Question"]))
            if original is None:
                raise KeyError(f"{path}: no RQ1 match for {row['Question']!r}")
            for field in RESULT_FIELDS:
                row[field] = original.get(field, "")
            row["stability_error"] = "FILLED_FROM_RQ1"
            filled += 1
        if len(rows) != len(source_rows):
            raise ValueError(f"{path}: expected {len(source_rows)} rows, found {len(rows)}")
        if any(not row.get("final_answer_ra", "").strip() for row in rows):
            raise ValueError(f"{path}: unresolved blank final answers remain")
        write(path, fields, rows)
        print(f"{model}: {path.name}: filled {filled} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()
    for model in args.models:
        fill_model(model)


if __name__ == "__main__":
    main()
