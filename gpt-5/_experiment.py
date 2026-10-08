"""Compatibility launcher for shared experiment implementations."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.launcher import launch


def run(module: str, legacy_script: str) -> None:
    launch(module, legacy_script)
