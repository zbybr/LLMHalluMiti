"""Shared QA generator, preserving the original command-line interface."""

from pathlib import Path
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    runpy.run_module("qa.generate_base_responses", run_name="__main__")
