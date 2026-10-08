#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python src/main.py --dataset_path "${1:?Provide the input dataset CSV}" --model_key "gpt-4o"
