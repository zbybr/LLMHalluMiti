"""Re-audit MutRepair stability outputs and calculate per-run metrics.

Base-response labels are inherited only when the sampled base response belongs
to the same question in the already-audited full MutRepair result. Final answers
are independently judged against the benchmark references. Existing labels are
preserved in ``recheck_hallucination_original`` before any update.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import audit_qa_labels as audit


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "gpt-4o": {
        "full": ROOT / "gpt-4o/outputs/gpt-4o_mutation_outputs_gpt-4o_dataset20251225_utf8_responses.csv",
        "directory": ROOT / "gpt-4o/outputs/stability",
    },
    "gpt-5": {
        "full": ROOT / "gpt-5/outputs/gpt-5_mutation_outputs_gpt-5_dataset20251225_utf8_responses.csv",
        "directory": ROOT / "gpt-5/outputs/stability",
    },
    "gemini": {
        "full": ROOT / "gemini/outputs/gemini_mutation_outputs_gemini_dataset20251225_utf8_responses.csv",
        "directory": ROOT / "gemini/outputs/stability",
    },
    "qwen3": {
        "full": ROOT / "qwen3/outputs/qwen3_32b_mutation_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed_newlines.csv",
        "directory": ROOT / "qwen3/outputs/stability",
    },
}


def question_key(question: str) -> str:
    normalized = " ".join(question.split())
    try:
        return normalized.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return normalized


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def insert_after(fieldnames: list[str], anchor: str, field: str) -> None:
    if field in fieldnames:
        return
    if anchor in fieldnames:
        fieldnames.insert(fieldnames.index(anchor) + 1, field)
    else:
        fieldnames.append(field)


def base_labels(full_path: Path) -> dict[str, str]:
    _, rows = read_rows(full_path)
    labels: dict[str, str] = {}
    for row in rows:
        label = row.get("is_hallucination", "").strip().upper()
        if label not in {"YES", "NO"}:
            raise ValueError(f"invalid full-data base label in {full_path}: {label!r}")
        labels[question_key(row["Question"])] = label
    return labels


def calculate(rows: list[dict[str, str]], final_column: str) -> dict[str, float | int]:
    total = len(rows)
    base_yes = sum(row["is_hallucination"].strip().upper() == "YES" for row in rows)
    final_yes = sum(row[final_column].strip().upper() == "YES" for row in rows)
    repaired = sum(
        row["is_hallucination"].strip().upper() == "YES"
        and row[final_column].strip().upper() == "NO"
        for row in rows
    )
    overcorrected = sum(
        row["is_hallucination"].strip().upper() == "NO"
        and row[final_column].strip().upper() == "YES"
        for row in rows
    )
    return {
        "samples": total,
        "base_hallucinations": base_yes,
        "final_hallucinations": final_yes,
        "repaired": repaired,
        "overcorrected": overcorrected,
        "HRR": repaired / base_yes * 100 if base_yes else math.nan,
        "RHR": final_yes / total * 100 if total else math.nan,
        "OCR": overcorrected / (total - base_yes) * 100 if total > base_yes else math.nan,
    }


def audit_file(
    *,
    model: str,
    path: Path,
    inherited_base_labels: dict[str, str],
    client: audit.JudgeClient,
    prompt: str,
    batch_size: int,
    workers: int,
    confidence_threshold: float,
    cache_root: Path,
) -> dict[str, object]:
    fieldnames, rows = read_rows(path)
    if len(rows) != 180:
        raise ValueError(f"{path}: expected 180 rows, found {len(rows)}")
    existing_base_complete = all(
        row.get("is_hallucination", "").strip().upper() in {"YES", "NO"}
        for row in rows
    )
    for row in rows:
        key = question_key(row["Question"])
        if key not in inherited_base_labels:
            raise KeyError(f"{path}: no audited base label for {row['Question']!r}")
        if not existing_base_complete:
            row["is_hallucination"] = inherited_base_labels[key]
        row["final_answer"] = row["final_answer_ra"]

    run = path.stem.rsplit("_", 1)[-1]
    cache_dir = cache_root / model / f"run_{run}"
    indices = list(range(len(rows)))
    pass1 = audit.run_pass(
        name="pass1",
        indices=indices,
        rows=rows,
        client=client,
        prompt=prompt,
        batch_size=batch_size,
        workers=workers,
        cache_path=cache_dir / "pass1.jsonl",
    )
    review_ids = [
        index
        for index in indices
        if pass1[index]["label"] == "YES"
        or float(pass1[index]["confidence"]) < confidence_threshold
    ]
    prior2 = {index: [pass1[index]] for index in review_ids}
    pass2 = audit.run_pass(
        name="pass2",
        indices=review_ids,
        rows=rows,
        client=client,
        prompt=prompt + "\n\nIndependently re-audit the item; the prior judgment is evidence only.",
        batch_size=batch_size,
        workers=workers,
        cache_path=cache_dir / "pass2.jsonl",
        prior_by_id=prior2,
    )
    disagreement_ids = [
        index for index in review_ids if pass1[index]["label"] != pass2[index]["label"]
    ]
    prior3 = {index: [pass1[index], pass2[index]] for index in disagreement_ids}
    pass3 = audit.run_pass(
        name="pass3",
        indices=disagreement_ids,
        rows=rows,
        client=client,
        prompt=audit.TIEBREAK_PROMPT + "\n\n" + prompt,
        batch_size=batch_size,
        workers=workers,
        cache_path=cache_dir / "pass3.jsonl",
        prior_by_id=prior3,
    )

    labels: dict[int, str] = {}
    for index in indices:
        if index in pass3:
            labels[index] = pass3[index]["label"]
        elif index in pass2:
            labels[index] = pass2[index]["label"]
        else:
            labels[index] = pass1[index]["label"]

    prior_valid = all(
        row.get("recheck_hallucination", "").strip().upper() in {"YES", "NO"}
        for row in rows
    )
    disagreements = 0
    for index, row in enumerate(rows):
        original = row.get("recheck_hallucination", "").strip().upper()
        if prior_valid:
            row["recheck_hallucination_original"] = original
            disagreements += original != labels[index]
        row["recheck_hallucination_reaudit"] = labels[index]
        row.pop("final_answer", None)

    insert_after(fieldnames, "base_response", "is_hallucination")
    insert_after(fieldnames, "final_answer_ra", "recheck_hallucination_reaudit")
    if "recheck_hallucination" in fieldnames:
        insert_after(fieldnames, "recheck_hallucination", "recheck_hallucination_original")
    audit.write_rows(path, fieldnames, rows)
    result: dict[str, object] = {
        "model": model,
        "run": int(run),
        "file": str(path.relative_to(ROOT)),
        "judge_model": client.model,
        "profile": "compatibility",
        "second_pass_samples": len(review_ids),
        "tiebreak_samples": len(disagreement_ids),
        "had_complete_original_labels": prior_valid,
        "disagreements_with_original": disagreements if prior_valid else "",
    }
    result.update(calculate(rows, "recheck_hallucination_reaudit"))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def restore_gpt4o_original_labels() -> None:
    """Undo an interrupted re-audit using untouched run-2 base labels."""
    directory = CONFIGS["gpt-4o"]["directory"]
    source_path = sorted(directory.glob("*.csv"))[1]
    _, source_rows = read_rows(source_path)
    source_base = {
        question_key(row["Question"]): row["is_hallucination"].strip().upper()
        for row in source_rows
    }
    for path in sorted(directory.glob("*.csv")):
        fieldnames, rows = read_rows(path)
        if "recheck_hallucination_original" not in fieldnames:
            continue
        for row in rows:
            row["is_hallucination"] = source_base[question_key(row["Question"])]
            original = row.get("recheck_hallucination_original", "").strip().upper()
            if original in {"YES", "NO"}:
                row["recheck_hallucination"] = original
            row.pop("recheck_hallucination_original", None)
        fieldnames = [name for name in fieldnames if name != "recheck_hallucination_original"]
        audit.write_rows(path, fieldnames, rows)
        print(f"restored original labels: {path}", flush=True)


def remove_provisional_reaudit_columns() -> None:
    provisional = {"recheck_hallucination_reaudit", "recheck_hallucination_original"}
    for config in CONFIGS.values():
        for path in sorted(config["directory"].glob("*.csv")):
            fieldnames, rows = read_rows(path)
            found = provisional.intersection(fieldnames)
            if not found:
                continue
            for row in rows:
                for name in provisional:
                    row.pop(name, None)
            fieldnames = [name for name in fieldnames if name not in provisional]
            audit.write_rows(path, fieldnames, rows)
            print(f"removed provisional columns from {path}", flush=True)


def summarize(results: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for model in CONFIGS:
        selected = [row for row in results if row["model"] == model]
        if not selected:
            continue
        record: dict[str, object] = {"model": model, "runs": len(selected)}
        for metric in ("RHR", "HRR", "OCR"):
            values = [float(row[metric]) for row in selected]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(record)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-gpt4o-original", action="store_true")
    parser.add_argument("--remove-provisional-columns", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--judge-model", default="claude-fable-5")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "stat/.stability_reaudit")
    parser.add_argument("--output", type=Path, default=ROOT / "stat/stability_reaudit_results.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "stat/stability_reaudit_summary.csv")
    args = parser.parse_args()

    if args.restore_gpt4o_original:
        restore_gpt4o_original_labels()
        return
    if args.remove_provisional_columns:
        remove_provisional_reaudit_columns()
        return

    env = audit.load_env(args.env)
    client = audit.JudgeClient(env["OPENAI_BASE_URL"], env["OPENAI_API_KEY"], args.judge_model)
    results: list[dict[str, object]] = []
    for model in args.models:
        config = CONFIGS[model]
        inherited = base_labels(config["full"])
        for path in sorted(config["directory"].glob("*.csv")):
            results.append(
                audit_file(
                    model=model,
                    path=path,
                    inherited_base_labels=inherited,
                    client=client,
                    prompt=audit.COMPATIBILITY_PROMPT,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    confidence_threshold=args.confidence_threshold,
                    cache_root=args.cache_dir,
                )
            )
    write_csv(args.output, results)
    summary = summarize(results)
    write_csv(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
