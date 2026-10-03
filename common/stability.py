"""Shared, resumable full-pipeline runner for MutRepair stability experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import random
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from common import prompts
from common.models import ModelSpec, spec_for_directory
from common.config import ROOT_ENV, read_env


ROOT = Path(__file__).resolve().parents[1]
RESULT_FIELDS = ["final_answer_ra", "token_cost_ra", "time_cost_ra", "mutation_list", "answer_list"]
DEFAULT_DATASETS = {
    "gpt-4o": "gpt-4o_dataset20251225_utf8_responses.csv",
    "gpt-5": "gpt-5_dataset20251225_utf8_responses.csv",
    "gemini": "gemini_dataset20251225_utf8_responses.csv",
    "qwen3": "qwen3_32b_dataset20251225_utf8_sig_responses_fixed.csv",
}


class QuotaLimitError(RuntimeError):
    """Raised after the provider repeats a quota cooldown across two windows."""


class ContentFilterError(RuntimeError):
    """Raised when the provider rejects a benchmark prompt before generation."""


def load_env(path: Path) -> dict[str, str]:
    values = read_env(path)
    missing = [name for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if not values.get(name)]
    if missing:
        raise ValueError(f"missing {', '.join(missing)} in {path}")
    return values


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"Question", "Answer", "base_response"}
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return fields, rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)
    os.replace(temporary, path)


class ChatClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.context = ssl.create_default_context()
        self._cooldown_lock = threading.Lock()
        self._next_request_at = 0.0
        self._first_quota_at: float | None = None
        self._fatal_quota_error: str | None = None

    def _wait_for_cooldown(self) -> None:
        while True:
            with self._cooldown_lock:
                if self._fatal_quota_error:
                    raise QuotaLimitError(self._fatal_quota_error)
                remaining = self._next_request_at - time.time()
            if remaining <= 0:
                return
            time.sleep(min(15.0, remaining))

    def _extend_cooldown(self, seconds: float) -> None:
        with self._cooldown_lock:
            self._next_request_at = max(self._next_request_at, time.time() + seconds)

    def call(self, spec: ModelSpec, messages: list[dict[str, str]], max_retries: int = 20) -> tuple[str, int]:
        payload: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "temperature": 0.1,
            **(spec.extra_body or {}),
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(max_retries):
            self._wait_for_cooldown()
            request = urllib.request.Request(
                self.url,
                data=encoded,
                method="POST",
                headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=300, context=self.context) as response:
                    body = json.load(response)
                content = body["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise ValueError("empty model response")
                usage = body.get("usage") or {}
                return content.strip(), int(usage.get("total_tokens") or 0)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(f"HTTP {error.code}: {detail}")
                if error.code == 400 and "content_filter" in detail:
                    raise ContentFilterError(detail) from error
                if error.code in {401, 402, 403}:
                    break
                cooldown = re.search(r"wait\s+(\d+)\s+seconds", detail, flags=re.IGNORECASE)
                if error.code == 429 and cooldown:
                    seconds = float(cooldown.group(1)) + 2.0
                    now = time.time()
                    with self._cooldown_lock:
                        if self._first_quota_at is None:
                            self._first_quota_at = now
                        elif now - self._first_quota_at >= 60.0:
                            self._fatal_quota_error = (
                                "provider repeated the same quota cooldown after a full wait window"
                            )
                            raise QuotaLimitError(self._fatal_quota_error)
                    self._extend_cooldown(seconds)
                    print(f"provider cooldown: waiting {seconds:.0f}s across all workers", flush=True)
            except Exception as error:  # providers expose several transport error types
                last_error = error
            if attempt + 1 < max_retries:
                wait = min(30.0, 1.5 * (attempt + 1)) + random.uniform(0, 1)
                print(f"retry {attempt + 1}/{max_retries}: {last_error}; wait {wait:.1f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"model call failed after {max_retries} attempts: {last_error}")


def extract_mutations(text: str) -> list[str]:
    matches = re.findall(r"(?ms)^\s*\d+[.)]\s*(.*?)(?=^\s*\d+[.)]\s*|\Z)", text)
    return [" ".join(item.split()).strip() for item in matches if item.strip()]


def generate_mutations(client: ChatClient, spec: ModelSpec, question: str, base: str) -> tuple[list[str], int]:
    # Resumed historical runs must not silently switch generation operators.
    snapshot = ROOT / spec.directory / "outputs/stability/.cache/mutation_prompt_snapshot.json"
    if snapshot.exists():
        template = json.loads(snapshot.read_text(encoding="utf-8"))["MUTATION_PROMPT"]
        mutation_prompt = template.format(n=5)
    else:
        mutation_prompt = prompts.qa_mutation_prompt(n=5)
    total_tokens = 0
    for attempt in range(3):
        text, used = client.call(
            spec,
            [
                {"role": "system", "content": mutation_prompt},
                {"role": "user", "content": f"Question: {question}\nBase_response: {base}"},
            ],
        )
        total_tokens += used
        mutations = extract_mutations(text)
        if len(mutations) == 5:
            return mutations, total_tokens
        print(f"mutation format retry {attempt + 1}/3: expected 5, received {len(mutations)}", flush=True)
    raise ValueError("model did not return exactly five numbered mutations")


def repair_one(client: ChatClient, spec: ModelSpec, row: dict[str, str]) -> dict[str, str]:
    started = time.time()
    question, base = row["Question"], row["base_response"]
    mutations, tokens = generate_mutations(client, spec, question, base)
    candidates = [*mutations, base]
    answers: list[str] = []
    for candidate in candidates:
        answer, used = client.call(
            spec,
            [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\nBase_response: {candidate}"},
            ],
        )
        tokens += used
        answers.append(answer.strip())

    answer_text = "\n".join(answers)
    ranking, used = client.call(
        spec,
        [
            {"role": "system", "content": prompts.RANKING_PROMPT},
            {"role": "user", "content": f"Question: {question}\nAnswers: {answer_text}"},
        ],
    )
    tokens += used
    final_answer, used = client.call(
        spec,
        [
            {"role": "system", "content": prompts.REFINE_PROMPT},
            {
                "role": "user",
                "content": f"SELECTION OUTPUT:\n{ranking}\n\nORIGINAL CANDIDATES:\n{answer_text}",
            },
        ],
    )
    tokens += used
    result = dict(row)
    result.update(
        {
            "final_answer_ra": final_answer.strip(),
            "token_cost_ra": str(tokens),
            "time_cost_ra": f"{time.time() - started:.6f}",
            "mutation_list": "\n".join(candidates),
            "answer_list": answer_text,
        }
    )
    return result


def load_cache(path: Path) -> dict[int, dict[str, str]]:
    cached: dict[int, dict[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    cached[int(item["index"])] = item["row"]
    return cached


def append_cache(path: Path, index: int, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps({"index": index, "row": row}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_error(path: Path, index: int, row: dict[str, str], error: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "index": index,
        "Question": row["Question"],
        "error_type": type(error).__name__,
        "error": str(error),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def output_path_for(spec: ModelSpec, dataset: Path, output_dir: Path, run: int) -> Path:
    return output_dir / f"{spec.output_prefix}_mutation_outputs_{dataset.stem}_full_run_{run}.csv"


def run_once(client: ChatClient, spec: ModelSpec, dataset: Path, output: Path, cache: Path, run: int, workers: int) -> None:
    input_fields, input_rows = read_rows(dataset)
    cached = load_cache(cache)
    if output.exists():
        output_fields, output_rows = read_rows(output)
        if len(output_rows) == len(input_rows) and set(RESULT_FIELDS).issubset(output_fields):
            for index, row in enumerate(output_rows):
                if row.get("final_answer_ra", "").strip():
                    cached[index] = row

    filtered: dict[int, str] = {}
    error_path = cache.with_suffix(".errors.jsonl")
    if error_path.exists():
        with error_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                index = int(record["index"])
                if record.get("error_type") != "ContentFilterError" or index in cached:
                    continue
                if not 0 <= index < len(input_rows) or record.get("Question") != input_rows[index]["Question"]:
                    raise ValueError(f"run {run}: content-filter cache does not match dataset index {index}")
                filtered[index] = str(record.get("error", "CONTENT_FILTERED"))
    pending = [(index, row) for index, row in enumerate(input_rows) if index not in cached and index not in filtered]
    print(
        f"run {run}: total={len(input_rows)}, completed={len(cached)}, pending={len(pending)}, workers={workers}",
        flush=True,
    )
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(repair_one, client, spec, row): index for index, row in pending}
            for done, future in enumerate(as_completed(futures), len(cached) + 1):
                index = futures[future]
                try:
                    row = future.result()
                except ContentFilterError as error:
                    filtered[index] = str(error)
                    append_error(error_path, index, input_rows[index], error)
                    print(
                        f"run {run}: content-filtered index {index}; continuing "
                        f"({done}/{len(input_rows)})",
                        flush=True,
                    )
                    continue
                cached[index] = row
                append_cache(cache, index, row)
                print(f"run {run}: completed {done}/{len(input_rows)}", flush=True)

    ordered: list[dict[str, str]] = []
    for index, input_row in enumerate(input_rows):
        if index in cached:
            row = cached[index]
            row["stability_error"] = ""
        elif index in filtered:
            row = dict(input_row)
            row.update({field: "" for field in RESULT_FIELDS})
            row["stability_error"] = "CONTENT_FILTERED"
        else:
            raise RuntimeError(f"run {run}: index {index} has neither a result nor a recorded error")
        ordered.append(row)
    output_fields = [
        *input_fields,
        *(field for field in RESULT_FIELDS if field not in input_fields),
        "stability_error",
    ]
    write_rows(output, output_fields, ordered)
    print(f"run {run}: wrote {output}", flush=True)


def main_for_model(model_directory: str) -> None:
    spec = spec_for_directory(model_directory)
    model_dir = ROOT / model_directory
    default_dataset = model_dir / "datasets" / DEFAULT_DATASETS[model_directory]
    parser = argparse.ArgumentParser(description="Run full-pipeline MutRepair stability experiments.")
    parser.add_argument("--dataset_path", type=Path, default=default_dataset)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--output_dir", type=Path, default=model_dir / "outputs" / "stability")
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--env", type=Path, default=ROOT_ENV)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if len(set(args.runs)) != len(args.runs) or any(run < 1 for run in args.runs):
        parser.error("--runs must contain unique positive integers")
    if args.output_path and len(args.runs) != 1:
        parser.error("--output_path can only be used with one run")

    dataset = args.dataset_path.resolve()
    _, rows = read_rows(dataset)
    outputs = [
        args.output_path.resolve() if args.output_path else output_path_for(spec, dataset, args.output_dir.resolve(), run)
        for run in args.runs
    ]
    print(f"model={spec.model}", flush=True)
    print(f"dataset={dataset} ({len(rows)} rows)", flush=True)
    print("mode=full-pipeline", flush=True)
    for run, output in zip(args.runs, outputs):
        print(f"run {run} -> {output}", flush=True)
    if args.dry_run:
        return

    env = load_env(args.env.resolve())
    client = ChatClient(env["OPENAI_BASE_URL"], env["OPENAI_API_KEY"])
    cache_root = args.output_dir.resolve() / ".cache" / dataset.stem
    for run, output in zip(args.runs, outputs):
        cache = cache_root / f"{spec.output_prefix}_full_run_{run}.jsonl"
        try:
            run_once(client, spec, dataset, output, cache, run, args.workers)
        except QuotaLimitError as error:
            marker = args.output_dir.resolve() / "stability_quota_blocked.json"
            marker.write_text(
                json.dumps(
                    {
                        "model": spec.model,
                        "dataset": str(dataset),
                        "run": run,
                        "reason": str(error),
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"quota limitation recorded in {marker}", flush=True)
            raise SystemExit(75) from error
