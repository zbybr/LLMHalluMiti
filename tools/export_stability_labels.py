"""Export existing stability labels without rejudging or modifying source data.

IDs are the one-based row positions in the original GPT-4o RQ1 full dataset.
Every model and run is aligned by normalized question, not by CSV row order.
YES/Y hallucination labels become fail; NO/N/NN labels become pass.
RHR, HRR and OCR are percentages recorded only in the first data row.
ocr is 1 exactly for a base pass followed by a repaired fail.
hrr is 1 exactly for a base fail followed by a repaired fail in run tables.
Initially correct responses are excluded from this repair-failure flag.
Manual-review and RQ1-fill flags remain in the source records; the exports
are label snapshots, not a claim that flagged cases are resolved.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

from audit_stability_with_knowledge import CONFIGS, question_key, read_rows


FIELDS = ["id", "base", "mutrepair", "ocr", "RHR", "HRR", "OCR"]
RUN_FIELDS = ["id", "base", "mutrepair", "hrr", "ocr", "RHR", "HRR", "OCR"]
LABELS = {
    "YES": "fail", "Y": "fail", "FAIL": "fail",
    "NO": "pass", "N": "pass", "NN": "pass", "PASS": "pass",
}


def pass_fail(value: str) -> str:
    label = value.strip().upper()
    if label not in LABELS:
        raise ValueError(f"Missing or unsupported correctness label: {value!r}")
    return LABELS[label]


def metrics(rows: list[dict[str, str | int]]) -> dict[str, str]:
    """Calculate percentages from the exact pass/fail labels being exported."""
    total = len(rows)
    initially_failed = sum(row["base"] == "fail" for row in rows)
    initially_passed = total - initially_failed
    final_failed = sum(row["mutrepair"] == "fail" for row in rows)
    repaired = sum(row["base"] == "fail" and row["mutrepair"] == "pass" for row in rows)
    overcorrected = sum(row["base"] == "pass" and row["mutrepair"] == "fail" for row in rows)

    def percentage(numerator: int, denominator: int) -> str:
        return f"{100 * numerator / denominator:.2f}" if denominator else "n.a."

    return {
        "RHR": percentage(final_failed, total),
        "HRR": percentage(repaired, initially_failed),
        "OCR": percentage(overcorrected, initially_passed),
    }


def export_one(source: Path, target: Path, ids: dict[str, int], *, include_flags: bool = True) -> int:
    is_run = re.fullmatch(r"mutrepair_run_\d+\.csv", target.name) is not None
    output_fields = RUN_FIELDS if is_run else FIELDS
    if not include_flags:
        output_fields = [field for field in output_fields if field not in {"hrr", "ocr", "manu"}]
    fields, rows = read_rows(source)
    final_field = (
        "recheck_hallucination" if "recheck_hallucination" in fields
        else "recheck_hallucination_ra"
    )
    if final_field not in fields:
        print(f"Pending audit, not exported: {source}")
        return 0
    if any(not row.get(final_field, "").strip() for row in rows):
        print(f"Incomplete labels, not exported: {source}")
        return 0
    keys = [question_key(row["Question"]) for row in rows]
    if len(keys) != len(ids) or len(set(keys)) != len(ids) or set(keys) != set(ids):
        raise ValueError(f"Incomplete, duplicated or mismatched question set: {source}")
    exported = sorted(
        (
            {
                "id": ids[key],
                "base": pass_fail(row["is_hallucination"]),
                "mutrepair": pass_fail(row[final_field]),
                "ocr": str(int(
                    pass_fail(row["is_hallucination"]) == "pass"
                    and pass_fail(row[final_field]) == "fail"
                )),
            }
            for key, row in zip(keys, rows)
        ),
        key=lambda row: row["id"],
    )
    if is_run and include_flags:
        for row in exported:
            row["hrr"] = str(int(row["base"] == "fail" and row["mutrepair"] == "fail"))
    summary = metrics(exported)
    exported[0].update(summary)
    if not include_flags:
        for row in exported:
            row.pop("ocr", None)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(exported)
    temporary.replace(target)
    with target.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        saved = list(reader)
        if reader.fieldnames != output_fields or len(saved) != len(exported):
            raise ValueError(f"Export verification failed: {target}")
        if any(
            str(expected["id"]) != actual["id"]
            or expected["base"] != actual["base"]
            or expected["mutrepair"] != actual["mutrepair"]
            or (include_flags and expected["ocr"] != actual["ocr"])
            or (is_run and include_flags and expected["hrr"] != actual["hrr"])
            for expected, actual in zip(exported, saved)
        ):
            raise ValueError(f"Saved labels differ from source: {target}")
        if any(saved[0][field] != value for field, value in summary.items()):
            raise ValueError(f"Saved metrics differ from labels: {target}")
        if any(row[field] for row in saved[1:] for field in summary):
            raise ValueError(f"Metrics must only appear in the first data row: {target}")
    print(f"Exported {target}: {len(saved)} rows; {summary}")
    return 1


def add_hrr_flags(models: list[str]) -> None:
    """Mark unrepaired initial errors without re-exporting reviewed labels."""
    total = 0
    for model in models:
        directory = CONFIGS[model]["directory"].parent / "eval_stability"
        for path in sorted(directory.glob("mutrepair_run_*.csv")):
            if re.fullmatch(r"mutrepair_run_\d+\.csv", path.name) is None:
                continue
            fields, rows = read_rows(path)
            if len(rows) != 1826 or len({row["id"] for row in rows}) != 1826:
                raise ValueError(f"Incomplete or duplicated run: {path}")
            original = [dict(row) for row in rows]
            if "hrr" not in fields:
                fields.insert(fields.index("mutrepair") + 1, "hrr")
            for row in rows:
                base, repaired = pass_fail(row["base"]), pass_fail(row["mutrepair"])
                row["hrr"] = str(int(base == "fail" and repaired == "fail"))
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            saved_fields, saved = read_rows(temporary)
            if saved_fields != fields or len(saved) != len(original):
                raise ValueError(f"Flag export verification failed: {path}")
            for before, after in zip(original, saved):
                if any(after[key] != value for key, value in before.items() if key != "hrr"):
                    raise ValueError(f"Existing values changed: {path}")
                expected = str(int(pass_fail(after["base"]) == "fail" and pass_fail(after["mutrepair"]) == "fail"))
                if after["hrr"] != expected:
                    raise ValueError(f"Incorrect repair flag: {path}")
            temporary.replace(path)
            total += 1
            print(f"Updated hrr: {path}: {sum(row['hrr'] == '1' for row in rows)} unrepaired initial errors")
    print(f"Updated {total} run tables; _f tables, originals and existing values unchanged.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--add-hrr-flags", action="store_true",
                        help="Only add hrr flags to existing run tables; exclude _f and original tables.")
    parser.add_argument("--clean", action="store_true",
                        help="Export final tables without hrr, ocr or manual-review flag columns.")
    args = parser.parse_args()
    if args.add_hrr_flags:
        add_hrr_flags(args.models)
        return
    _, canonical = read_rows(CONFIGS["gpt-4o"]["full"])
    ids = {question_key(row["Question"]): index for index, row in enumerate(canonical, 1)}
    if len(ids) != len(canonical):
        raise ValueError("Canonical dataset contains duplicated normalized questions")
    total = 0
    for model in args.models:
        config = CONFIGS[model]
        destination = config["directory"].parent / "eval_stability"
        total += export_one(config["full"], destination / "mutrepair_original.csv", ids, include_flags=not args.clean)
        for source in sorted(config["directory"].glob("*_full_run_*.csv")):
            match = re.search(r"_full_run_(\d+)$", source.stem)
            if match is None:
                raise ValueError(f"Cannot determine run number: {source}")
            total += export_one(source, destination / f"mutrepair_run_{match[1]}.csv", ids, include_flags=not args.clean)
    print(f"Exported {total} CSV files; source answers and audit flags were not changed.")


if __name__ == "__main__":
    main()
