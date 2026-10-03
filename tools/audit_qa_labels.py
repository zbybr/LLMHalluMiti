"""Audit QA result labels with an OpenAI-compatible judge API.

The script is resumable and preserves the input CSV schema.  YES means the
candidate answer is hallucinated/incorrect; NO means it is acceptable.
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
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a strict factuality auditor for a QA benchmark.

For every item, judge the CANDIDATE ANSWER against the QUESTION and REFERENCE ANSWERS.
Return label YES when the candidate is hallucinated or incorrect, and label NO when it is acceptable.

Rules:
1. Semicolon-separated reference text contains alternative acceptable formulations or examples, not a checklist that the candidate must reproduce in full. Matching any clearly sufficient reference formulation is enough.
2. Use the references as the authority. Do not use outside knowledge to overturn them.
3. Accept faithful paraphrases, concise answers, and answers that give the correct core answer with harmless wording differences.
4. Return YES for a wrong entity, date, number, relationship, polarity, or other material contradiction.
5. Return YES when a genuinely required multi-part answer is materially incomplete, when the candidate evades a question that the reference answers, or when it presents a different answer. Do not demand exhaustive coverage unless the question explicitly asks for all items or an exact count.
6. If the reference says the answer is unknown, unanswerable, or based on a false premise, the candidate must preserve that uncertainty or correction.
7. The references are not an exhaustive world model. Extra details absent from the reference are not automatically errors, and a correct example or subset can answer an open-ended plural question. Ignore nonessential additions unless they clearly contradict the reference or change the core answer. Never label YES merely because an added entity is not listed in the reference.
8. Judge only the candidate answer, not the base answer or the repair method.
9. A concise correction such as "zero," "none," "never," or "there is no such X" can adequately reject a false premise even if it does not repeat the reference explanation.
10. If the candidate clearly says there is no scientific evidence or validated method, it may mention alleged beliefs, commonly used devices, or purported practices as context without endorsing them; do not label that contextualized answer YES.

Return valid JSON only in this exact shape:
{"results":[{"id":0,"label":"YES","confidence":0.95,"reason":"brief reason"}]}
Include every supplied id exactly once. Confidence must be a number from 0 to 1. Keep each reason under 30 words.
"""


REVIEW_PROMPT = """Independently re-audit these potentially difficult QA judgments.
Apply the same strict reference-based rules. The prior judgment is evidence only, not a target.
Pay special attention to semantic equivalence, negation, dates/numbers, false premises, and whether extra details materially contradict the reference.

Return valid JSON only in this exact shape:
{"results":[{"id":0,"label":"YES","confidence":0.95,"reason":"brief reason"}]}
Include every supplied id exactly once. Confidence must be a number from 0 to 1. Keep each reason under 30 words.
"""


TIEBREAK_PROMPT = """Resolve disagreements between two factuality audits.
Use the QUESTION and REFERENCE ANSWERS as the authority. YES means the candidate is hallucinated/incorrect; NO means acceptable.
Do not choose a label to match an expected aggregate. Decide each item on its own merits.

Return valid JSON only in this exact shape:
{"results":[{"id":0,"label":"YES","confidence":0.95,"reason":"brief reason"}]}
Include every supplied id exactly once. Confidence must be a number from 0 to 1. Keep each reason under 30 words.
"""


COMPATIBILITY_PROMPT = """You are auditing final answers in a QA benchmark using the repository's original correctness standard.

For every item, compare the CANDIDATE ANSWER with the QUESTION and REFERENCE ANSWERS.
Return label YES only when the candidate contains a clear, material factual error or directly contradicts the reference. Return label NO when the core answer is correct or acceptably compatible.

Rules:
1. Semicolon-separated references are alternative acceptable answers or examples, not a cumulative checklist.
2. Accept paraphrases, concise answers, reasonable correct examples, and non-exhaustive subsets unless the question explicitly requires all items or an exact count.
3. References are correctness anchors, not an exhaustive world model. Do not reject a candidate merely because it includes a plausible detail or entity absent from the references.
4. A clearly wrong entity, date, number, relation, polarity, or answer to a false premise is YES.
5. A concise "zero," "none," "never," or equivalent correction can reject a false premise without repeating the reference explanation.
6. Caveated descriptions of beliefs, alleged practices, or unvalidated devices are acceptable when the candidate clearly states that evidence or scientific validation is absent.
7. If the candidate is compatible with the reference and there is no clear material error, label NO. Do not turn ambiguity or harmless incompleteness into an error.
8. Judge only the final candidate answer. Do not use an expected aggregate result or the base label as a target.

Return valid JSON only in this exact shape:
{"results":[{"id":0,"label":"YES","confidence":0.95,"reason":"brief reason"}]}
Include every supplied id exactly once. Confidence must be a number from 0 to 1. Keep each reason under 30 words.
"""


LEGACY_PROMPT = """You are given a correct answer and another context. Judge whether the final answer in the context is correct according to the correct answer.

For each item, return label NO when the context is correct and label YES when it is incorrect or hallucinated. Accept semantic equivalence; exact wording is not required.

Return valid JSON only in this exact shape:
{"results":[{"id":0,"label":"YES","confidence":0.95,"reason":"brief reason"}]}
Include every supplied id exactly once. Confidence must be a number from 0 to 1. Keep each reason under 30 words.
"""


def load_env(path: Path) -> dict[str, str]:
    return read_env(path)


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


class JudgeClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.ssl_context = ssl.create_default_context()

    def judge(self, system_prompt: str, items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        expected_ids = {int(item["id"]) for item in items}
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
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
                with urllib.request.urlopen(
                    request, timeout=180, context=self.ssl_context
                ) as response:
                    body = json.load(response)
                content = body["choices"][0]["message"]["content"]
                parsed = extract_json(content)
                results: dict[int, dict[str, Any]] = {}
                for result in parsed["results"]:
                    item_id = int(result["id"])
                    label = str(result["label"]).strip().upper()
                    confidence = float(result["confidence"])
                    if label not in {"YES", "NO"}:
                        raise ValueError(f"invalid label {label!r}")
                    if not 0 <= confidence <= 1:
                        raise ValueError(f"invalid confidence {confidence}")
                    results[item_id] = {
                        "label": label,
                        "confidence": confidence,
                        "reason": str(result.get("reason", "")).strip(),
                    }
                if set(results) != expected_ids:
                    raise ValueError(
                        f"judge returned ids {sorted(results)}; expected {sorted(expected_ids)}"
                    )
                return results
            except Exception as error:  # network and response validation
                last_error = error
                if attempt == 7:
                    break
                time.sleep(min(20, 2 ** attempt))
        raise RuntimeError(f"judge request failed after retries: {last_error}")


def load_cache(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                cache[int(item["id"])] = item
    return cache


def append_cache(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_item(index: int, row: dict[str, str], prior: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": index,
        "question": row["Question"],
        "reference_answers": row["Answer"],
        "candidate_answer": row["final_answer"],
    }
    if prior:
        item["prior_judgments"] = prior
    return item


def run_pass(
    *,
    name: str,
    indices: list[int],
    rows: list[dict[str, str]],
    client: JudgeClient,
    prompt: str,
    batch_size: int,
    workers: int,
    cache_path: Path,
    prior_by_id: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[int, dict[str, Any]]:
    cache = load_cache(cache_path)
    pending = [index for index in indices if index not in cache]
    batches = [pending[offset : offset + batch_size] for offset in range(0, len(pending), batch_size)]
    total_batches = len(batches)

    def judge_batch(batch_number: int, batch_ids: list[int]) -> tuple[int, list[dict[str, Any]]]:
        items = [
            make_item(
                index,
                rows[index],
                None if prior_by_id is None else prior_by_id.get(index),
            )
            for index in batch_ids
        ]
        results = client.judge(prompt, items)
        records = [
            {"id": index, "pass": name, **results[index]} for index in batch_ids
        ]
        return batch_number, records

    if workers == 1:
        completed = (
            judge_batch(batch_number, batch_ids)
            for batch_number, batch_ids in enumerate(batches, 1)
        )
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [
            executor.submit(judge_batch, batch_number, batch_ids)
            for batch_number, batch_ids in enumerate(batches, 1)
        ]
        completed = (future.result() for future in as_completed(futures))

    try:
        for batch_number, records in completed:
            append_cache(cache_path, records)
            cache.update({int(record["id"]): record for record in records})
            yes_count = sum(record["label"] == "YES" for record in records)
            print(
                f"{name}: batch {batch_number}/{total_batches}, "
                f"items={len(records)}, YES={yes_count}",
                flush=True,
            )
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return cache


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".audit-tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def metrics(rows: list[dict[str, str]], labels: dict[int, str]) -> dict[str, float | int]:
    total = len(rows)
    base_yes = sum(row["is_hallucination"].strip().upper() == "YES" for row in rows)
    final_yes = sum(labels[index] == "YES" for index in range(total))
    repaired = sum(
        row["is_hallucination"].strip().upper() == "YES" and labels[index] == "NO"
        for index, row in enumerate(rows)
    )
    overcorrected = sum(
        row["is_hallucination"].strip().upper() == "NO" and labels[index] == "YES"
        for index, row in enumerate(rows)
    )
    return {
        "samples": total,
        "base_hallucinations": base_yes,
        "final_hallucinations": final_yes,
        "HRR_pct": repaired / base_yes * 100 if base_yes else 0,
        "RHR_pct": final_yes / total * 100 if total else 0,
        "OCR_pct": overcorrected / (total - base_yes) * 100 if total > base_yes else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env", type=Path, default=ROOT_ENV)
    parser.add_argument("--model", default="claude-fable-5")
    parser.add_argument(
        "--profile",
        choices=("strict", "compatibility", "legacy"),
        default="strict",
    )
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=Path("stat/.gpt5_drhall_audit"))
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=3622)
    parser.add_argument("--write-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = load_env(args.env)
    client = JudgeClient(env["OPENAI_BASE_URL"], env["OPENAI_API_KEY"], args.model)
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"Question", "Answer", "final_answer", "is_hallucination", "recheck_hallucination"}
    missing = required - set(fieldnames)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    indices = list(range(len(rows)))
    if args.limit is not None:
        random.Random(args.seed).shuffle(indices)
        indices = sorted(indices[: args.limit])

    primary_prompt = {
        "strict": SYSTEM_PROMPT,
        "compatibility": COMPATIBILITY_PROMPT,
        "legacy": LEGACY_PROMPT,
    }[args.profile]
    review_prompt = primary_prompt + "\n\nIndependently re-audit the item; the prior judgment is evidence only."
    tiebreak_prompt = TIEBREAK_PROMPT + "\n\n" + primary_prompt

    pass1 = run_pass(
        name="pass1",
        indices=indices,
        rows=rows,
        client=client,
        prompt=primary_prompt,
        batch_size=args.batch_size,
        workers=args.workers,
        cache_path=args.cache_dir / "pass1.jsonl",
    )
    review_ids = [
        index
        for index in indices
        if pass1[index]["label"] == "YES"
        or float(pass1[index]["confidence"]) < args.confidence_threshold
    ]
    prior2 = {index: [pass1[index]] for index in review_ids}
    pass2 = run_pass(
        name="pass2",
        indices=review_ids,
        rows=rows,
        client=client,
        prompt=review_prompt,
        batch_size=args.batch_size,
        workers=args.workers,
        cache_path=args.cache_dir / "pass2.jsonl",
        prior_by_id=prior2,
    )
    disagreement_ids = [
        index for index in review_ids if pass1[index]["label"] != pass2[index]["label"]
    ]
    prior3 = {index: [pass1[index], pass2[index]] for index in disagreement_ids}
    pass3 = run_pass(
        name="pass3",
        indices=disagreement_ids,
        rows=rows,
        client=client,
        prompt=tiebreak_prompt,
        batch_size=args.batch_size,
        workers=args.workers,
        cache_path=args.cache_dir / "pass3.jsonl",
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

    summary = {
        "model": args.model,
        "profile": args.profile,
        "audited_samples": len(indices),
        "second_pass_samples": len(review_ids),
        "tiebreak_samples": len(disagreement_ids),
        "YES": sum(label == "YES" for label in labels.values()),
        "NO": sum(label == "NO" for label in labels.values()),
    }
    if len(indices) == len(rows):
        summary.update(metrics(rows, labels))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.write_labels:
        if len(indices) != len(rows):
            raise ValueError("--write-labels requires auditing the complete input")
        for index, row in enumerate(rows):
            row["recheck_hallucination"] = labels[index]
        output_path = args.output or args.input
        write_rows(output_path, fieldnames, rows)
        print(f"wrote labels to {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted; cached completed batches can be resumed", file=sys.stderr)
        raise
