"""Resolve the invoking model directory and start the shared stability runner."""

import os
from pathlib import Path

from common.stability import main_for_model


main_for_model(Path(os.environ["LLM_MODEL_DIR"]).name)
