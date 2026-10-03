"""Single source of truth for model-specific experiment settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    directory: str
    model: str
    output_prefix: str
    extra_body: dict[str, Any] | None = None

    def model_dir(self, repo_root: Path) -> Path:
        return repo_root / self.directory


MODEL_SPECS = {
    "gpt-4o": ModelSpec("gpt-4o", "gpt-4o", "gpt-4o"),
    "gpt-5": ModelSpec("gpt-5", "gpt-5", "gpt-5"),
    "gemini": ModelSpec(
        "gemini", "gemini-2.5-flash-thinking", "gemini"
    ),
    "qwen3": ModelSpec(
        "qwen3", "qwen3-32b", "qwen3-32b", {"enable_thinking": False}
    ),
}


def spec_for_directory(directory: str) -> ModelSpec:
    try:
        return MODEL_SPECS[directory]
    except KeyError as error:
        raise ValueError(f"unknown model directory: {directory}") from error
