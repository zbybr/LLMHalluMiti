"""Audit stability outputs with a fallible-gold, knowledge-based protocol.

The judge evaluates both the base response and repaired answer. Reference
answers are evidence rather than ground truth; questionable references are
flagged explicitly for later human inspection.
"""

from __future__ import annotations

from pathlib import Path as _ConfigPath
import sys as _config_sys

_config_root = next(
    parent for parent in _ConfigPath(__file__).resolve().parents
    if (parent / "common" / "config.py").is_file()
)
if str(_config_root) not in _config_sys.path:
    _config_sys.path.insert(0, str(_config_root))
from common.config import ROOT_ENV, load_root_env, read_env

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import ssl
import statistics
import time
import unicodedata
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DATE = "2025-12-25"
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

SYSTEM_PROMPT = f"""You are a careful factuality adjudicator for a research benchmark.

For each item, independently judge BASE ANSWER and FINAL ANSWER. Label YES when
the answer is materially incorrect, hallucinated, contradictory, or fails to
answer the question; label NO when it is acceptable.

The benchmark snapshot date is {SNAPSHOT_DATE}. Judge time-sensitive wording
such as "current" as of that date, not today's date.

REFERENCE ANSWERS ARE FALLIBLE EVIDENCE, NOT ABSOLUTE GROUND TRUTH. They may be
incomplete, malformed, ambiguous, stale, or factually wrong. Use your own
well-established knowledge and reasoning. When a reference conflicts with a
fact you know reliably, do not force the answer to match the reference; flag
the reference status instead. If your knowledge is uncertain and the reference
is plausible, use it as evidence and lower confidence.

Judgment rules:
1. Accept faithful paraphrases, concise direct answers, and independently
   correct alternatives absent from the reference.
2. Semicolon-separated references are usually alternatives, not a checklist.
3. For open-ended plural questions, a correct representative subset can be
   acceptable unless the question explicitly requires all items or an exact count.
4. Reject wrong entities, dates, numbers, relations, polarity, fabricated
   details, and evasive non-answers. "I don't know" is not equivalent to
   explicitly explaining that no consensus exists.
5. Treat descriptions of religious, cultural, or alleged beliefs as
   descriptions; do not confuse them with empirical endorsement.
6. For ambiguous or underspecified questions, accept a reasonable factual
   interpretation unless the answer asserts a clear falsehood.
7. Judge each answer independently. Never choose labels to match an expected
   aggregate, paper table, repair direction, or previous label.

Set gold_status to one of:
- OK: reference is usable and substantially correct;
- INCOMPLETE: correct but materially non-exhaustive;
- INCORRECT: factually wrong or malformed in a way that changes the answer;
- OUTDATED: correct for another time but not the snapshot date;
- AMBIGUOUS: question/reference admits multiple defensible interpretations.

Return valid JSON only:
{{"results":[{{"id":0,"base_label":"YES","final_label":"NO","confidence":0.95,"gold_status":"OK","needs_manual_review":false,"reason":"brief evidence-based reason"}}]}}
Include every supplied id exactly once. Keep each reason under 45 words.
"""

REVIEW_PROMPT = SYSTEM_PROMPT + """

This is an independent second adjudication of difficult cases. Prior judgments
are context, not a target. Reconsider factual knowledge, snapshot timing,
question ambiguity, and whether the reference itself is unreliable.
"""

TIEBREAK_PROMPT = SYSTEM_PROMPT + """

Resolve the disagreement between prior adjudications. Decide from the question,
answers, snapshot date, and factual knowledge. Do not average or defer to either
prior label, and do not optimize for any expected paper result.
"""


def load_env(path: Path) -> dict[str, str]:
    return read_env(path)


def repair_mojibake(value: str) -> str:
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def question_key(question: str) -> str:
    value = repair_mojibake(" ".join(question.split()))
    value = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    return value.casefold()


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


class PairJudge:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.context = ssl.create_default_context()

    def judge(self, prompt: str, items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        expected = {int(item["id"]) for item in items}
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(8):
            request = urllib.request.Request(
                self.url,
                data=encoded,
                method="POST",
                headers={
                    "Authorization": "Bearer " + self.api_key,
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=300, context=self.context) as response:
                    body = json.load(response)
                content = body["choices"][0]["message"]["content"]
                parsed = extract_json(content)
                results: dict[int, dict[str, Any]] = {}
                for raw in parsed["results"]:
                    item_id = int(raw["id"])
                    base_label = str(raw["base_label"]).strip().upper()
                    final_label = str(raw["final_label"]).strip().upper()
                    confidence = float(raw["confidence"])
                    gold_status = str(raw["gold_status"]).strip().upper()
                    if base_label not in {"YES", "NO"} or final_label not in {"YES", "NO"}:
                        raise ValueError("invalid YES/NO label")
                    if not 0 <= confidence <= 1:
                        raise ValueError("confidence outside [0, 1]")
                    if gold_status not in {"OK", "INCOMPLETE", "INCORRECT", "OUTDATED", "AMBIGUOUS"}:
                        raise ValueError(f"invalid gold_status {gold_status!r}")
                    results[item_id] = {
                        "id": item_id,
                        "base_label": base_label,
                        "final_label": final_label,
                        "confidence": confidence,
                        "gold_status": gold_status,
                        "needs_manual_review": bool(raw.get("needs_manual_review", False)),
                        "reason": str(raw.get("reason", "")).strip(),
                    }
                if set(results) != expected:
                    raise ValueError(f"returned ids {sorted(results)}; expected {sorted(expected)}")
                return results
            except Exception as error:
                last_error = error
                if attempt == 7:
                    break
                time.sleep(min(20, 2 ** attempt))
        raise RuntimeError(f"judge failed after retries: {last_error}")


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".knowledge-audit-tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_cache(path: Path) -> dict[int, dict[str, Any]]:
    cached: dict[int, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    cached[int(item["id"])] = item
    return cached


def append_cache(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_item(index: int, row: dict[str, str], prior: list[dict[str, Any]] | None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": index,
        "question": row["Question"],
        "reference_answers": row["Answer"],
        "base_answer": row["base_response"],
        "final_answer": row["final_answer_ra"],
    }
    if prior:
        item["prior_judgments"] = prior
    return item


def run_pass(
    *,
    name: str,
    ids: list[int],
    rows: list[dict[str, str]],
    judge: PairJudge,
    prompt: str,
    batch_size: int,
    workers: int,
    cache_path: Path,
    prior: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[int, dict[str, Any]]:
    cache = load_cache(cache_path)
    pending = [item_id for item_id in ids if item_id not in cache]
    batches = [pending[offset : offset + batch_size] for offset in range(0, len(pending), batch_size)]

    def execute(batch_number: int, batch: list[int]) -> tuple[int, list[dict[str, Any]]]:
        items = [make_item(item_id, rows[item_id], None if prior is None else prior.get(item_id)) for item_id in batch]
        results = judge.judge(prompt, items)
        return batch_number, [{"pass": name, **results[item_id]} for item_id in batch]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(execute, number, batch) for number, batch in enumerate(batches, 1)]
        for future in as_completed(futures):
            number, records = future.result()
            append_cache(cache_path, records)
            cache.update({int(record["id"]): record for record in records})
            print(f"{name}: batch {number}/{len(batches)}, items={len(records)}", flush=True)
    return cache


def source_base_labels(path: Path) -> dict[str, str]:
    _, rows = read_rows(path)
    return {
        question_key(row["Question"]): row["is_hallucination"].strip().upper()
        for row in rows
    }


def metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    total = len(rows)
    base_yes = sum(row["is_hallucination"] == "YES" for row in rows)
    final_yes = sum(row["recheck_hallucination"] == "YES" for row in rows)
    repaired = sum(row["is_hallucination"] == "YES" and row["recheck_hallucination"] == "NO" for row in rows)
    overcorrected = sum(row["is_hallucination"] == "NO" and row["recheck_hallucination"] == "YES" for row in rows)
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


def insert_after(fields: list[str], anchor: str, name: str) -> None:
    if name not in fields:
        fields.insert(fields.index(anchor) + 1 if anchor in fields else len(fields), name)


def audit_file(
    model: str,
    path: Path,
    source_labels: dict[str, str],
    judge: PairJudge,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fields, rows = read_rows(path)
    if len(rows) != 1826:
        raise ValueError(f"{path}: expected 1826 rows, found {len(rows)}")
    match = re.search(r"_full_run_(\d+)$", path.stem)
    if not match:
        raise ValueError(f"cannot determine full-run number from {path.name}")
    run = int(match.group(1))
    cache = args.cache_dir / model / f"run_{run}"
    identity = {
        "judge_model": judge.model,
        "snapshot_date": SNAPSHOT_DATE,
        "confidence_threshold": args.confidence_threshold,
        "inputs_sha256": hashlib.sha256(json.dumps(
            [make_item(index, row, None) for index, row in enumerate(rows)],
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest(),
        "prompts_sha256": hashlib.sha256((SYSTEM_PROMPT + REVIEW_PROMPT + TIEBREAK_PROMPT).encode("utf-8")).hexdigest(),
    }
    manifest = cache / "input_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Audit cache inputs or judge settings changed: {cache}")
    else:
        if any(cache.glob("pass*.jsonl")):
            raise ValueError(f"Unverified legacy audit cache; use a separate cache directory: {cache}")
        recovered: list[dict[str, Any]] = []
        if args.recover_legacy_pass1_from is not None:
            old_path = args.recover_legacy_pass1_from / model / f"run_{run}" / "pass1.jsonl"
            records = [json.loads(line) for line in old_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            # The old 180 sampled judgments precede this full-input audit.
            # Only the subsequently appended full-input judgments are reused.
            if {int(record["id"]) for record in records[:180]} != set(range(180)):
                raise ValueError(f"Cannot identify the known 180-item legacy prefix: {old_path}")
            if any(not 180 <= int(record["id"]) < len(rows) for record in records[180:]):
                raise ValueError(f"Unexpected IDs after the legacy prefix: {old_path}")
            recovered = list({int(record["id"]): record for record in records[180:]}.values())
        cache.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        if recovered:
            append_cache(cache / "pass1.jsonl", recovered)
            print(f"{model} run {run}: recovered {len(recovered)} valid full-input judgments; excluded 180 legacy sampled judgments", flush=True)
    ids = list(range(len(rows)))
    pass1 = run_pass(
        name="pass1", ids=ids, rows=rows, judge=judge, prompt=SYSTEM_PROMPT,
        batch_size=args.batch_size, workers=args.workers, cache_path=cache / "pass1.jsonl",
    )
    review_ids = [
        item_id for item_id in ids
        if float(pass1[item_id]["confidence"]) < args.confidence_threshold
        or pass1[item_id]["gold_status"] != "OK"
        or bool(pass1[item_id]["needs_manual_review"])
        or pass1[item_id]["base_label"] != pass1[item_id]["final_label"]
    ]
    prior2 = {item_id: [pass1[item_id]] for item_id in review_ids}
    pass2 = run_pass(
        name="pass2", ids=review_ids, rows=rows, judge=judge, prompt=REVIEW_PROMPT,
        batch_size=args.batch_size, workers=args.workers, cache_path=cache / "pass2.jsonl", prior=prior2,
    )
    disagreement_ids = [
        item_id for item_id in review_ids
        if (pass1[item_id]["base_label"], pass1[item_id]["final_label"])
        != (pass2[item_id]["base_label"], pass2[item_id]["final_label"])
    ]
    prior3 = {item_id: [pass1[item_id], pass2[item_id]] for item_id in disagreement_ids}
    pass3 = run_pass(
        name="pass3", ids=disagreement_ids, rows=rows, judge=judge, prompt=TIEBREAK_PROMPT,
        batch_size=args.batch_size, workers=args.workers, cache_path=cache / "pass3.jsonl", prior=prior3,
    )

    full_label_mismatches = 0
    flagged_gold = 0
    manual = 0
    for item_id, row in enumerate(rows):
        decision = pass3.get(item_id) or pass2.get(item_id) or pass1[item_id]
        source = source_labels.get(question_key(row["Question"]), "")
        row["is_hallucination_full_dataset"] = source
        row["is_hallucination"] = decision["base_label"]
        row["recheck_hallucination"] = decision["final_label"]
        row["audit_confidence"] = str(decision["confidence"])
        row["gold_status"] = decision["gold_status"]
        row["audit_needs_manual_review"] = str(bool(decision["needs_manual_review"])).upper()
        row["audit_reason"] = decision["reason"]
        full_label_mismatches += source in {"YES", "NO"} and source != decision["base_label"]
        flagged_gold += decision["gold_status"] != "OK"
        manual += bool(decision["needs_manual_review"])

    insert_after(fields, "base_response", "is_hallucination_full_dataset")
    insert_after(fields, "is_hallucination_full_dataset", "is_hallucination")
    insert_after(fields, "final_answer_ra", "recheck_hallucination")
    insert_after(fields, "recheck_hallucination", "audit_confidence")
    insert_after(fields, "audit_confidence", "gold_status")
    insert_after(fields, "gold_status", "audit_needs_manual_review")
    insert_after(fields, "audit_needs_manual_review", "audit_reason")
    write_rows(path, fields, rows)
    result: dict[str, Any] = {
        "model": model,
        "run": run,
        "file": str(path.relative_to(ROOT)),
        "judge_model": judge.model,
        "second_pass_samples": len(review_ids),
        "tiebreak_samples": len(disagreement_ids),
        "base_label_mismatches_vs_full": full_label_mismatches,
        "flagged_gold_samples": flagged_gold,
        "manual_review_samples": manual,
    }
    result.update(metrics(rows))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for model in dict.fromkeys(row["model"] for row in results):
        selected = [row for row in results if row["model"] == model]
        record: dict[str, Any] = {"model": model, "runs": len(selected)}
        for metric in ("RHR", "HRR", "OCR"):
            values = [float(row[metric]) for row in selected]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_std"] = statistics.stdev(values)
        summary.append(record)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--judge-model", default="gpt-5.6-sol")
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent audit batches within each run.")
    parser.add_argument("--parallel-runs", type=int, default=3,
                        help="Runs audited simultaneously; total concurrency is workers times parallel-runs.")
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "stat/.stability_knowledge_audit")
    parser.add_argument("--recover-legacy-pass1-from", type=Path,
                        help="Recover newly appended full-input pass1 records after a verified 180-item legacy prefix into a fresh cache.")
    parser.add_argument("--results", type=Path, default=ROOT / "stat/stability_knowledge_results.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "stat/stability_knowledge_summary.csv")
    args = parser.parse_args()
    if args.workers < 1 or args.parallel_runs < 1 or args.batch_size < 1:
        parser.error("workers, parallel-runs, and batch-size must all be positive")

    env = load_env(args.env)
    judge = PairJudge(env["OPENAI_BASE_URL"], env["OPENAI_API_KEY"], args.judge_model)
    results: list[dict[str, Any]] = []
    jobs: list[tuple[str, Path, dict[str, str]]] = []
    for model in args.models:
        config = CONFIGS[model]
        labels = source_base_labels(config["full"])
        paths = sorted(config["directory"].glob("*_full_run_*.csv"))
        if len(paths) != 3:
            raise ValueError(f"{model}: expected three full-run CSVs, found {len(paths)}")
        for path in paths:
            jobs.append((model, path, labels))
    print(
        f"audit concurrency: {args.parallel_runs} runs x {args.workers} workers "
        f"= at most {args.parallel_runs * args.workers} concurrent requests",
        flush=True,
    )
    # Each run owns its CSV and model/run-specific cache; summary files are
    # written only by this coordinating thread after every run has finished.
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = [
            executor.submit(audit_file, model, path, labels, judge, args)
            for model, path, labels in jobs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (args.models.index(row["model"]), int(row["run"])))
    write_table(args.results, results)
    summary = summarize(results)
    write_table(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
