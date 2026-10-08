"""Launch a shared pipeline while preserving legacy script command lines."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

from .models import spec_for_directory
from .config import ROOT_ENV


def launch(module: str, legacy_script: str) -> None:
    script = Path(legacy_script).resolve()
    model_dir = script.parent
    if model_dir.name == "tools":
        model_dir = model_dir.parent
    spec = spec_for_directory(model_dir.name)

    os.environ["LLM_MODEL_KEY"] = spec.model
    os.environ["LLM_OUTPUT_PREFIX"] = spec.output_prefix
    os.environ["LLM_MODEL_DIR"] = str(model_dir)
    os.environ["LLM_ENV_FILE"] = str(ROOT_ENV)
    os.environ["LLM_EXTRA_BODY_JSON"] = json.dumps(spec.extra_body or {})
    runpy.run_module(module, run_name="__main__")
