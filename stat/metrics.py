"""Read-only metrics extracted from the experiment finalization scripts.

Empty labels always mean an incorrect answer. Unknown nonempty labels are
errors, not missing data. CSV inputs are never rewritten or re-adjudicated.
All reported rates use percent units; a zero denominator is reported as None.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path
from typing import Any, Iterable


csv.field_size_limit(64_000_000)


def display_path(path: Path) -> str:
    """Keep machine-specific absolute paths out of exported reports."""
    try:
        return path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        return path.name


def parse_success(value: Any, semantics: str) -> int:
    """Return 1=correct, 0=incorrect; semantics describe the binary column."""
    if semantics not in {"correct", "incorrect"}:
        raise ValueError(f"unknown outcome semantics: {semantics}")
    normalized = "" if value is None else str(value).strip().lower()
    if not normalized:
        return 0
    positive = {"1", "true", "yes", "y"}
    negative = {"0", "false", "no", "n", "nn"}
    # Literal correctness labels keep their meaning regardless of numeric coding.
    if normalized in {"correct", "pass", "passed"}:
        return 1
    if normalized in {"incorrect", "fail", "failed"}:
        return 0
    if normalized in positive:
        indicator = 1
    elif normalized in negative:
        indicator = 0
    else:
        raise ValueError(f"cannot parse binary outcome value: {value!r}")
    return indicator if semantics == "correct" else 1 - indicator


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    # Keep already-damaged answer text untouched; only label columns are parsed.
    with path.open(encoding="utf-8-sig", errors="surrogateescape", newline="") as f:
        reader = csv.DictReader(f)
        fields = [str(k).strip().lstrip("\ufeff") for k in reader.fieldnames or []]
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"missing or duplicate headers: {path}")
        rows = []
        for number, row in enumerate(reader, 1):
            if None in row:
                raise ValueError(f"extra CSV fields in record {number}: {path}")
            rows.append({k: row[original] for k, original in zip(fields, reader.fieldnames)})
    if not rows:
        raise ValueError(f"result file is empty: {path}")
    return fields, rows


def infer_result_columns(path: Path) -> dict[str, str]:
    fields, _ = read_rows(path)
    headers = set(fields)
    if "is_hallucination" in headers:
        candidates = ["recheck_hallucination_ra", "recheck_hallucination"]
        candidates += sorted(k for k in headers if k.startswith("recheck_hallucination"))
        final = next((k for k in candidates if k in headers), None)
        if final is None:
            raise ValueError(f"no rechecked hallucination column found in {path}")
        return dict(schema="qa", id_column="Question", base_column="is_hallucination",
                    base_semantics="incorrect", final_column=final)
    if "base_pass" in headers:
        candidates = sorted(k for k in headers if k.endswith("_final_pass"))
        if not candidates:
            candidates = sorted(k for k in headers if k.endswith("_pass") and k != "base_pass")
        if len(candidates) != 1:
            raise ValueError(f"expected one final pass column in {path}; found {candidates}")
        identity = "task_id" if "task_id" in headers else "id"
        return dict(schema="code", id_column=identity, base_column="base_pass",
                    base_semantics="correct", final_column=candidates[0])
    if {"id", "base", "mutrepair"} <= headers:
        return dict(schema="stability", id_column="id", base_column="base",
                    base_semantics="correct", final_column="mutrepair")
    raise ValueError(f"unrecognised result schema: {path}")


def metric_counts(base: list[int], final: list[int]) -> dict[str, Any]:
    """The HRR/RHR/OCR count formulas used in the source finalization scripts."""
    if not base or len(base) != len(final):
        raise ValueError("nonempty, equally sized base/final outcomes are required")
    if any(v not in (0, 1) for v in base + final):
        raise ValueError("outcomes must be binary correctness values")
    total = len(base)
    base_wrong = base.count(0)
    final_wrong = final.count(0)
    repaired = sum(b == 0 and a == 1 for b, a in zip(base, final))
    overcorrected = sum(b == 1 and a == 0 for b, a in zip(base, final))
    return {
        "samples": total, "base_hallucinations": base_wrong,
        "final_hallucinations": final_wrong, "repaired": repaired,
        "overcorrected": overcorrected,
        "HR": 100 * base_wrong / total,
        "RHR": 100 * final_wrong / total,
        "HRR": 100 * repaired / base_wrong if base_wrong else None,
        "OCR": 100 * overcorrected / (total - base_wrong) if total != base_wrong else None,
    }


def question_key(value: str) -> str:
    """Match legacy spelling/encoding variants in memory, without editing CSVs."""
    for _ in range(3):
        try:
            decoded = value.encode("latin-1").decode("utf-8")
        except UnicodeError:
            break
        if decoded == value:
            break
        value = decoded
    value = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
    return " ".join(value.split())


def metrics_for_file(path: Path, *, final_column: str | None = None,
                     questions: set[str] | None = None) -> dict[str, Any]:
    columns = infer_result_columns(path)
    fields, rows = read_rows(path)
    if questions is not None:
        rows = [row for row in rows if question_key(row.get("Question", "")) in questions]
        if len(rows) != len(questions):
            raise ValueError(f"dataset membership does not match result questions: {path}")
    base_column = columns["base_column"]
    final_column = final_column or columns["final_column"]
    identity = columns["id_column"]
    if not {identity, base_column, final_column} <= set(fields):
        raise ValueError(f"missing ID or label columns in {path}")
    ids = [(row.get(identity) or "").strip() for row in rows]
    if identity == "Question":
        ids = [question_key(key) for key in ids]
    if any(not key for key in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"empty or duplicate IDs in {path}")
    semantics = columns["base_semantics"]
    base = [parse_success(row[base_column], semantics) for row in rows]
    final = [parse_success(row[final_column], semantics) for row in rows]
    return {
        "schema": columns["schema"], "final_column": final_column,
        **metric_counts(base, final),
        "empty_base_labels": sum(not (row[base_column] or "").strip() for row in rows),
        "empty_final_labels": sum(not (row[final_column] or "").strip() for row in rows),
        "empty_label_policy": "incorrect",
    }


def write_table(path: Path, rows: list[dict[str, Any]], *, input_paths: Iterable[Path] = ()) -> None:
    """Write a summary, never overwrite a source result or a model output tree."""
    if not rows:
        raise ValueError("cannot write an empty summary")
    if path.suffix.lower() != ".csv":
        raise ValueError("summary destination must be a .csv file")
    target = path.resolve()
    if target in {p.resolve() for p in input_paths} or "outputs" in target.parts:
        raise ValueError(f"summary destination must not overwrite model outputs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
