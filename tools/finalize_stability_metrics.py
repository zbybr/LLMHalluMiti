"""Harmonize base labels across repeated runs and export stability metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
from typing import Any

from audit_stability_with_knowledge import CONFIGS, question_key, read_rows, write_rows


ROOT = Path(__file__).resolve().parents[1]


def load_manual_decisions() -> dict[tuple[str, int, str], dict[str, Any]]:
    """Read explicitly confirmed human decisions; never infer them from flags."""
    path = ROOT / "stat/stability_manual_decisions.json"
    if not path.exists():
        return {}
    decisions = json.loads(path.read_text(encoding="utf-8"))["decisions"]
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for decision in decisions:
        key = (decision["model"], int(decision["run"]), question_key(decision["Question"]))
        if key in result or decision["final_label"] not in {"YES", "NO"}:
            raise ValueError(f"Duplicate or invalid manual decision: {key}")
        result[key] = decision
    return result


def apply_manual_decision(row: dict[str, str], decision: dict[str, Any]) -> None:
    """Keep the earlier automatic label alongside the human adjudication."""
    if not row.get("audit_previous_final_label"):
        row["audit_previous_final_label"] = row["recheck_hallucination"]
    row["recheck_hallucination"] = decision["final_label"]
    row["audit_needs_manual_review"] = "FALSE"
    row["audit_manual_review_status"] = decision["decision_source"]


def metric_row(model: str, run: str | int, path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    base_yes = sum(row["is_hallucination"] == "YES" for row in rows)
    final_field = "recheck_hallucination" if "recheck_hallucination" in rows[0] else "recheck_hallucination_ra"
    final_yes = sum(row[final_field] == "YES" for row in rows)
    repaired = sum(row["is_hallucination"] == "YES" and row[final_field] == "NO" for row in rows)
    overcorrected = sum(row["is_hallucination"] == "NO" and row[final_field] == "YES" for row in rows)
    return {
        "model": model,
        "run": run,
        "file": str(path.relative_to(ROOT)),
        "samples": total,
        "base_hallucinations": base_yes,
        "final_hallucinations": final_yes,
        "repaired": repaired,
        "overcorrected": overcorrected,
        "HRR": repaired / base_yes * 100,
        "RHR": final_yes / total * 100,
        "OCR": overcorrected / (total - base_yes) * 100,
        "flagged_gold_samples": sum(row.get("gold_status", "OK") != "OK" for row in rows),
        "manual_review_samples": sum(row.get("audit_needs_manual_review", "FALSE") == "TRUE" for row in rows),
    }


def write_table(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def import_reviewed_labels(models: list[str]) -> None:
    """Import explicit human-reviewed _f tables by canonical ID, not row order."""
    manifest_path = ROOT / "stat/stability_manual_decisions.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"decisions": []}
    decisions = load_manual_decisions()
    _, canonical = read_rows(CONFIGS["gpt-4o"]["full"])
    questions = {index: question_key(row["Question"]) for index, row in enumerate(canonical, 1)}
    if len(questions) != 1826 or len(set(questions.values())) != 1826:
        raise ValueError("Canonical IDs must identify 1826 unique questions")
    imported = changed = 0
    for model in models:
        config = CONFIGS[model]
        _, original = read_rows(config["full"])
        base_labels = {question_key(row["Question"]): row["is_hallucination"] for row in original}
        for run in (1, 2, 3):
            reviewed_path = config["directory"].parent / "eval_stability" / f"mutrepair_run_{run}_f.csv"
            _, reviewed = read_rows(reviewed_path)
            ids = [int(row["id"]) for row in reviewed]
            if len(ids) != 1826 or len(set(ids)) != 1826 or set(ids) != set(questions):
                raise ValueError(f"Incomplete or duplicate reviewed IDs: {reviewed_path}")
            paths = list(config["directory"].glob(f"*_full_run_{run}.csv"))
            if len(paths) != 1:
                raise ValueError(f"Expected one source for {model} run {run}")
            _, source_rows = read_rows(paths[0])
            source = {question_key(row["Question"]): row for row in source_rows}
            if len(source_rows) != 1826 or set(source) != set(questions.values()):
                raise ValueError(f"Source questions do not match reviewed IDs: {paths[0]}")
            for reviewed_row in reviewed:
                item_id = int(reviewed_row["id"])
                key = questions[item_id]
                for column in ("base", "mutrepair"):
                    if reviewed_row[column] not in {"pass", "fail"}:
                        raise ValueError(f"Invalid reviewed {column}: {reviewed_path}, id {item_id}")
                base_label = "YES" if reviewed_row["base"] == "fail" else "NO"
                if base_label != base_labels[key]:
                    raise ValueError(f"Reviewed base label differs from RQ1: {reviewed_path}, id {item_id}")
                final_label = "YES" if reviewed_row["mutrepair"] == "fail" else "NO"
                row = source[key]
                decision_key = (model, run, key)
                prior = decisions.get(decision_key)
                decision = dict(prior) if prior else {
                    "model": model, "run": run, "id": item_id, "Question": row["Question"],
                    "previous_final_label": row.get("audit_previous_final_label") or row["recheck_hallucination"],
                }
                if prior and prior["final_label"] != final_label:
                    decision["previous_human_label"] = prior["final_label"]
                decision.update({
                    "final_label": final_label,
                    "decision_source": "USER_REVIEWED_CSV",
                    "review_file": str(reviewed_path.relative_to(ROOT)),
                })
                decisions[decision_key] = decision
                imported += 1
                changed += row["recheck_hallucination"] != final_label
    manifest["decision_basis"] = (
        "Explicit user-confirmed manual decisions, including labels imported from reviewed _f CSVs. "
        "These decisions override automated labels only for the recorded model, run and question."
    )
    manifest["decisions"] = list(decisions.values())
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Imported {imported} reviewed labels; {changed} differ from current full-run labels.")


def revoke_flagged_pass(models: list[str]) -> None:
    """Withdraw user-confirmed flagged-pass overrides and restore saved labels."""
    flagged_pass_sources = {"USER_CONFIRMED_FLAGGED_PASS", "USER_CONFIRMED_MANUAL_REVIEW"}
    manifest_path = ROOT / "stat/stability_manual_decisions.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [
        decision for decision in manifest["decisions"]
        if decision["model"] in models
        and decision["decision_source"] in flagged_pass_sources
    ]
    if not selected:
        raise ValueError("No flagged-pass decisions match the requested models")
    pending = []
    matched = set()
    for model in models:
        for run in (1, 2, 3):
            paths = list(CONFIGS[model]["directory"].glob(f"*_full_run_{run}.csv"))
            if len(paths) != 1:
                raise ValueError(f"Expected one full source for {model} run {run}")
            fields, rows = read_rows(paths[0])
            lookup = {question_key(row["Question"]): row for row in rows}
            if len(rows) != 1826 or len(lookup) != 1826:
                raise ValueError(f"Invalid full source: {paths[0]}")
            for decision in selected:
                if decision["model"] != model or int(decision["run"]) != run:
                    continue
                key = question_key(decision["Question"])
                row = lookup[key]
                previous = decision["previous_final_label"]
                if (previous not in {"YES", "NO"}
                    or row.get("audit_previous_final_label") != previous
                    or row.get("audit_manual_review_status") != decision["decision_source"]
                    or row["recheck_hallucination"] != decision["final_label"]):
                    raise ValueError(f"Decision no longer matches source: {model}, {run}, {key}")
                row["recheck_hallucination"] = previous
                row["audit_needs_manual_review"] = "TRUE"
                row["audit_previous_final_label"] = ""
                row["audit_manual_review_status"] = ""
                matched.add((model, run, key))
            for field in ("audit_previous_final_label", "audit_manual_review_status"):
                if field in fields and not any(row.get(field) for row in rows):
                    fields.remove(field)
                    for row in rows:
                        row.pop(field, None)
            pending.append((paths[0], fields, rows))
    if len(matched) != len(selected):
        raise ValueError("Not all selected decisions matched the full sources")
    for path, fields, rows in pending:
        write_rows(path, fields, rows)
    manifest["decisions"] = [
        decision for decision in manifest["decisions"]
        if (decision["model"], int(decision["run"]), question_key(decision["Question"])) not in matched
    ]
    manifest.setdefault("revoked_decisions", []).extend(
        dict(decision, revocation_reason="User withdrew the pass confirmation to recheck these items.")
        for decision in selected
    )
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Restored {len(selected)} automatic labels and pending-review flags; overrides archived as revoked.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--import-reviewed-labels", nargs="+", choices=list(CONFIGS),
                        help="Import human-reviewed _f tables for these models before finalizing.")
    parser.add_argument("--revoke-flagged-pass", nargs="+", choices=list(CONFIGS),
                        help="Withdraw flagged-pass overrides for these models and restore automatic labels.")
    args = parser.parse_args()
    if args.revoke_flagged_pass:
        if args.import_reviewed_labels or not set(args.revoke_flagged_pass).issubset(args.models):
            parser.error("Revoked models must be selected and cannot also be imported")
        revoke_flagged_pass(args.revoke_flagged_pass)
    if args.import_reviewed_labels:
        if not set(args.import_reviewed_labels).issubset(args.models):
            parser.error("Imported models must also be included in --models")
        import_reviewed_labels(args.import_reviewed_labels)
    manual_decisions = load_manual_decisions()
    applied_decisions: set[tuple[str, int, str]] = set()
    results: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for model in args.models:
        config = CONFIGS[model]
        paths = sorted(config["directory"].glob("*_full_run_*.csv"))
        loaded = [(path, *read_rows(path)) for path in paths]
        if len(loaded) != 3:
            raise ValueError(f"{model}: expected 3 runs, found {len(loaded)}")
        _, original_rows = read_rows(config["full"])
        source_labels = {
            question_key(row["Question"]): row["is_hallucination"]
            for row in original_rows
        }
        if len(source_labels) != 1826:
            raise ValueError(f"{model}: original full run does not contain 1826 unique questions")
        results.append(metric_row(model, "Original", config["full"], original_rows))

        for path, fields, rows in loaded:
            match = re.search(r"_full_run_(\d+)$", path.stem)
            if not match:
                raise ValueError(f"cannot determine full-run number from {path.name}")
            run = int(match.group(1))
            if len(rows) != 1826:
                raise ValueError(f"{path}: expected 1826 rows, found {len(rows)}")
            for row in rows:
                row["is_hallucination"] = source_labels[question_key(row["Question"])]
                decision_key = (model, run, question_key(row["Question"]))
                if decision_key in manual_decisions:
                    apply_manual_decision(row, manual_decisions[decision_key])
                    applied_decisions.add(decision_key)
                    for field in ("audit_previous_final_label", "audit_manual_review_status"):
                        if field not in fields:
                            fields.append(field)
                if row["audit_needs_manual_review"] == "TRUE":
                    manual_rows.append(
                        {
                            "model": model,
                            "run": run,
                            "Question": row["Question"],
                            "Answer": row["Answer"],
                            "base_response": row["base_response"],
                            "final_answer_ra": row["final_answer_ra"],
                            "is_hallucination": row["is_hallucination"],
                            "recheck_hallucination": row["recheck_hallucination"],
                            "gold_status": row["gold_status"],
                            "audit_confidence": row["audit_confidence"],
                            "audit_reason": row["audit_reason"],
                        }
                    )
                if row["gold_status"] != "OK":
                    gold_rows.append(
                        {
                            "model": model,
                            "run": run,
                            "Question": row["Question"],
                            "Answer": row["Answer"],
                            "gold_status": row["gold_status"],
                            "audit_reason": row["audit_reason"],
                        }
                    )
            write_rows(path, fields, rows)
            results.append(metric_row(model, run, path, rows))

    expected_decisions = {key for key in manual_decisions if key[0] in args.models}
    if applied_decisions != expected_decisions:
        raise ValueError("Not all confirmed manual decisions matched the selected runs")
    summaries: list[dict[str, Any]] = []
    for model in args.models:
        selected = [row for row in results if row["model"] == model]
        summary: dict[str, Any] = {"model": model, "runs": len(selected)}
        for metric in ("RHR", "HRR", "OCR"):
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values)
        summaries.append(summary)

    blocked_exports: list[Path] = []
    output_tables = [
        (ROOT / "stat/stability_knowledge_results.csv", results),
        (ROOT / "stat/stability_knowledge_summary.csv", summaries),
        (ROOT / "stat/stability_manual_review.csv", manual_rows),
        (ROOT / "stat/stability_gold_issues.csv", gold_rows),
    ]
    for model in args.models:
        directory = CONFIGS[model]["directory"]
        output_tables.extend([
            (directory / "stability_results.csv", [row for row in results if row["model"] == model]),
            (directory / "stability_summary.csv", [row for row in summaries if row["model"] == model]),
        ])
    for destination, table in output_tables:
        try:
            empty_review_fields = [
                "model", "run", "Question", "Answer", "base_response", "final_answer_ra",
                "is_hallucination", "recheck_hallucination", "gold_status", "audit_confidence", "audit_reason",
            ] if destination.name == "stability_manual_review.csv" else None
            write_table(destination, table, empty_review_fields)
        except PermissionError:
            blocked_exports.append(destination)
            print(f"Close the application holding this file and rerun: {destination}")
    for summary in summaries:
        print(summary)
    print(f"manual-review rows: {len(manual_rows)}")
    print(f"confirmed human decisions applied: {len(applied_decisions)}")
    print(f"gold-issue rows: {len(gold_rows)}")
    if blocked_exports:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
