"""Restore the one QA stability sample missing from Gemini and Qwen3.

The canonical sample order is taken from the complete GPT-5 sampled file.
Each restored row uses that model's own base response from its full dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "gpt-5/datasets/gpt-5_dataset20251225_utf8_responses_sampled.csv"
TARGETS = {
    "gemini": (
        ROOT / "gemini/datasets/gemini_dataset20251225_utf8_responses.csv",
        ROOT / "gemini/datasets/gemini_dataset20251225_utf8_responses_sampled.csv",
    ),
    "qwen3": (
        ROOT / "qwen3/datasets/qwen3_32b_dataset20251225_utf8_sig_responses_fixed.csv",
        ROOT / "qwen3/datasets/qwen3_32b_dataset20251225_utf8_responses_sampled.csv",
    ),
}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def key(question: str) -> str:
    normalized = " ".join(question.split())
    # Some historical sampled files contain UTF-8 bytes decoded once as Latin-1
    # (for example, ``MichÃ¨le`` instead of ``Michèle``).
    try:
        repaired = normalized.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        repaired = normalized
    return repaired


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".restore-tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    _, canonical_rows = read_rows(CANONICAL)
    canonical_questions = [key(row["Question"]) for row in canonical_rows]
    if len(canonical_questions) != 180 or len(set(canonical_questions)) != 180:
        raise ValueError("canonical stability sample must contain 180 unique questions")

    for model, (full_path, sample_path) in TARGETS.items():
        fieldnames, sampled_rows = read_rows(sample_path)
        _, full_rows = read_rows(full_path)
        sampled_by_question = {key(row["Question"]): row for row in sampled_rows}
        full_by_question = {key(row["Question"]): row for row in full_rows}
        missing = [question for question in canonical_questions if question not in sampled_by_question]
        extra = [question for question in sampled_by_question if question not in set(canonical_questions)]
        if extra:
            raise ValueError(f"{model}: unexpected sampled questions: {extra}")
        if len(missing) > 1:
            raise ValueError(f"{model}: expected at most one missing question, found {len(missing)}")
        if missing:
            question = missing[0]
            if question not in full_by_question:
                raise KeyError(f"{model}: missing question is absent from full dataset: {question}")
            sampled_by_question[question] = {
                name: full_by_question[question].get(name, "") for name in fieldnames
            }
        restored = [sampled_by_question[question] for question in canonical_questions]
        if len(restored) != 180:
            raise AssertionError(f"{model}: restored row count is {len(restored)}")
        write_rows(sample_path, fieldnames, restored)
        print(f"{model}: restored {len(missing)} row; total={len(restored)}")


if __name__ == "__main__":
    main()
