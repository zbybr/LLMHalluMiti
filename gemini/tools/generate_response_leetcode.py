"""Model-specific base-response generation entry point."""

from pathlib import Path
import sys

MODEL_DIR = Path(__file__).resolve().parent.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from _experiment import run


if __name__ == "__main__":
    run("leetcode.generate_base_responses", __file__)
