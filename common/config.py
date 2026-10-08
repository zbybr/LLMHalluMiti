"""Repository-wide API configuration, independent of the working directory."""

from __future__ import annotations

import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_ENV = REPO_ROOT / ".env"


def _read_simple_env(path: Path) -> dict[str, str]:
    """Support simple .env files for dependency-free audit/stability tools."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid configuration line in {path}")
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if value.startswith(('"', "'")):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0:
                raise ValueError("Multiline .env values require python-dotenv")
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        value = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda match: values.get(match[1], os.getenv(match[1], "")),
            value,
        )
        values[name] = value
    return values


def read_env(path: Path | None = None) -> dict[str, str]:
    """Read root configuration; an explicit path is an optional CLI override."""
    selected = ROOT_ENV if path is None else Path(path).resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"Configuration file not found: {selected}")
    try:
        from dotenv import dotenv_values
    except ImportError:
        parsed = _read_simple_env(selected)
    else:
        parsed = dotenv_values(selected, encoding="utf-8-sig")
    values = {
        name: value.replace("\\:", ":")
        for name, value in parsed.items()
        if value is not None
    }
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OLLAMA_BASE_URL"):
        if not values.get(name) and os.getenv(name):
            values[name] = os.environ[name]
    return values


def load_root_env() -> bool:
    """Load only the root .env, with file values taking precedence."""
    values = read_env()
    os.environ.update(values)
    return bool(values)


def model_path(*parts: str) -> Path:
    """Resolve model-owned artifacts without changing the caller's directory."""
    return Path(os.getenv("LLM_MODEL_DIR", str(Path.cwd()))).joinpath(*parts)


class LazyOpenAIClient:
    """Read credentials only when an API operation is actually requested."""

    def __init__(self) -> None:
        self._client = None

    def __getattr__(self, name):
        if self._client is None:
            from openai import OpenAI

            load_root_env()
            self._client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        return getattr(self._client, name)
