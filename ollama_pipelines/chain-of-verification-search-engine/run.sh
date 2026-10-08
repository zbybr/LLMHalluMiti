#!/usr/bin/env bash
set -euo pipefail

dataset_path="${1:?Usage: bash run.sh DATASET_CSV [OLLAMA_MODEL]}"
model_key="${2:-qwen3:32b}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec python "$script_dir/src/main.py" --dataset_path "$dataset_path" --model_key "$model_key"
