# MutRepair: Mutation-Guided Hallucination Repair in LLMs

This repository contains the implementation and experimental scripts for the paper **"Mutation-Guided Hallucination Repair in LLMs"**.

MutRepair is a **fully self-contained** framework that actively *repairs* (not just detects) non-factual hallucinations in LLM outputs — requiring **no retrieval, external knowledge bases, tools, or fine-tuning**. Inspired by software engineering practices (metamorphic testing, fault injection, and automated program repair), it works with both open- and closed-source LLMs, on both **natural-language QA** and **code generation** tasks.

## How It Works

Given a question and the LLM's base response, MutRepair runs three stages:

1. **Mutation Construction** — Expand the base response into a set of metamorphic mutations (meaning-preserving rewrite, structural transformation, semantic polarity shift, and an additional algorithm/data-structure variant for code) to expose latent faults.
2. **Fault-Injected Self-Repair** — Inject an explicit fault assumption into each mutation ("this response may be faulty"), shifting the model from passive validation into active fault localization and repair, which counteracts confirmation bias.
3. **LLM-as-Judge Aggregation** — Rank the independently repaired candidates via pairwise comparison and select the best-ranked one as the final answer, without any external ground truth.

## Repository Structure

```
.
├── gpt-4o/                  # Experiments with GPT-4o
├── gpt-5/                   # Experiments with GPT-5
├── gemini/                  # Experiments with Gemini 2.5 Flash (Thinking)
├── qwen3/                   # Experiments with Qwen3-32B (via OpenAI-compatible API)
└── ollama_pipelines/        # Local Qwen3 pipelines served through Ollama
```

Each model directory follows the same layout:

```
<model>/
├── repair_with_mutation.py           # MutRepair pipeline for QA benchmarks
├── repair_with_mutation_leetcode.py  # MutRepair pipeline for LeetCode code generation
├── repair_with_mutation_nomr.py      # Ablation: mutations WITHOUT metamorphic relations
├── repair_with_mutation_stability*.py# Repeated-run stability experiments (RQ3)
├── auto_evaluation*.py               # Evaluation scripts (QA judge / LeetCode test execution)
├── drhall_leetcode.py                # DrHall baseline (metamorphic input variants)
├── chain-of-verification-main/       # CoVe baseline (LLM self-verification)
├── chain-of-verification-search-engine/  # CoVe-SE baseline (search-engine-augmented)
├── llm_prompts/prompts.py            # All prompt templates
├── tools/                            # Datasets for evaluation
├── tools/                            # Base-response generation & data utilities
└── outputs/                          # Generated repair results (CSV)
```

## Setup

**Requirements:** Python 3.10+, and the following packages:

```bash
pip install openai pandas python-dotenv tqdm tiktoken langchain
```

**API configuration:** create a `.env` file in the model directory you want to run:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-endpoint/v1   # any OpenAI-compatible endpoint
```

For the open-source model (Qwen3-32B), we use the official [Ollama](https://ollama.com) release; the `ollama_pipelines/` directory contains the corresponding pipelines.

## Usage

The workflow has three steps, using GPT-4o on a QA benchmark as an example:

**1. Generate base responses**

```bash
cd gpt-4o
python tools/generate_response.py --dataset_path ./datasets/xxx.csv
# → gpt-4o_truthfulqa_responses.csv  (adds a `base_response` column)
```

**2. Run the MutRepair repair pipeline**

```bash
# Natural-language QA (input CSV columns: Question, Answer, base_response)
python repair_with_mutation.py --dataset_path ./gpt-4o_truthfulqa_responses.csv \
    --model_key gpt-4o --n_mutations 5

# Code generation (input CSV columns: task_id, problem_description, starter_code, base_response)
python repair_with_mutation_leetcode.py --dataset_path ./gpt-4o_leetcode_responses.csv \
    --model_key gpt-4o --n_mutations 5
```

Results are written to `./outputs/<model>_mutrepair_<dataset>.csv` with the final repaired answer plus token/time cost per sample.

**3. Evaluate**

```bash
# LeetCode: executes each answer against the hidden unit tests
python auto_evaluation_leetcode.py --repair_paths ./outputs/xxx.csv --dataset_path ./datasets/xxx.csv
```

Evaluation reports three metrics: **RHR** (recheck hallucination rate, ↓), **HRR** (hallucination repair rate, ↑), and **OCR** (over-correction rate, ↓).

