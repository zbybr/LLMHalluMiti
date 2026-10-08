# MutRepair: Mutation-Guided Hallucination Repair in LLMs

This repository contains the implementation, datasets, experimental outputs, and statistical analysis scripts for **Mutation-Guided Hallucination Repair in LLMs**.

MutRepair is a self-contained inference-time framework for repairing factual and code hallucinations. Repair requires no retrieval, external knowledge, execution feedback, or model retraining. The experiments cover TruthfulQA, HotpotQA, FreshQA, and LeetCode across GPT-4o, GPT-5, Gemini, and Qwen3.

## How It Works

Given a question and an initial response, MutRepair applies three stages:

1. **Metamorphic Mutation Generation:** construct output-space mutations using Meaning-Preserving Rewrite, Structural Transformation, and Polarity Transformation. Polarity Transformation introduces or removes double negation without reversing the claim. Code additionally uses the Algorithm/Data-Structure Variant operator.
2. **Fault-Assumption-Guided Self-Repair:** independently re-examine the base response and each mutation under an explicit fault assumption, producing repaired candidates. Five mutations plus the base response give six repair trajectories by default.
3. **LLM-as-Judge Aggregation:** compare repaired candidates in the context of the original question and select the final output without an external oracle.

Prompt templates are maintained in `common/prompts.py`. QA uses prompt-guided pairwise ranking followed by answer extraction; the current LeetCode implementation requests a score matrix in one judge call and computes its row totals in Python.

## Repository Structure

```text
.
├── common/              # Shared prompts, model settings, configuration, and stability runner
├── qa/                  # Shared QA generation, repair, and baseline implementations
├── leetcode/            # Shared code generation, repair, baselines, and execution evaluation
├── gpt-4o/              # Model-specific entry points, datasets, and outputs
├── gpt-5/
├── gemini/
├── qwen3/
├── ollama_pipelines/    # Alternative local Qwen3 pipelines using Ollama
├── llm_prompts/         # Compatibility imports for the shared prompt templates
├── tools/               # Shared QA base-response generation entry point
├── stat/                # Metrics, paired McNemar tests, and stability summaries
├── test/                # Offline configuration, entry-point, and statistics tests
├── requirements.txt     # Pinned dependencies for this repository version
└── README.md
```

The four model directories retain the original script names as entry points to the shared implementations:

```text
<model>/
├── datasets/                               # Model-specific inputs with base responses
├── outputs/                                # Repair outputs and evaluation labels
│   ├── cot/                                # CoT results
│   ├── cove/                               # CoVe results
│   ├── cove-se/                            # Search-augmented CoVe results
│   ├── drhall/                             # DrHall results
│   ├── eval_code/                          # Per-task code correctness labels
│   ├── stability/                          # Repeated-run answers
│   └── eval_stability/                     # Original and three repeated-run label tables
├── tools/generate_response.py              # QA base-response generation
├── tools/generate_response_leetcode.py      # Code base-response generation
├── repair_with_mutation.py                 # MutRepair for QA
├── repair_with_mutation_leetcode.py         # MutRepair for code
├── repair_with_mutation_stability.py        # Full-pipeline QA stability runs
├── cot.py / cot_leetcode.py                 # CoT baselines
├── drhall.py / drhall_leetcode.py           # DrHall baselines
└── auto_evaluation_leetcode.py              # Execution-based code evaluation
```

Additional ablation and sensitivity entry points are included where available. The GPT-4o, GPT-5, and Gemini directories also contain `chain-of-verification-main/` (CoVe) and `chain-of-verification-search-engine/` (CoVe-SE); the local Qwen3 versions are under `ollama_pipelines/`.

## Setup

Use Python 3.11 or later; Python 3.12 is recommended. Run the following from the repository root, preferably in a virtual environment:

```bash
python -m pip install -r requirements.txt
```

The dependency pins apply to this repository version, not to the historical experiment environment.

For model calls, create **one `.env` file in the repository root**, not in individual model directories:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-endpoint/v1
```

All model entry points share these credentials. Configure an endpoint that provides the requested model before running its scripts. Model identifiers are defined in `common/models.py`:

| Directory | Default model identifier |
| --- | --- |
| `gpt-4o` | `gpt-4o` |
| `gpt-5` | `gpt-5` |
| `gemini` | `gemini-2.5-flash-thinking` |
| `qwen3` | `qwen3-32b` |

For `ollama_pipelines/`, also set `OLLAMA_BASE_URL=http://localhost:11434` in the root `.env` and provide a running Ollama server with the requested model. CoVe-SE requires network access for search. Credentials are not included in this repository and should never be committed.

## Recompute Metrics from Included Results

These commands use the existing labels. They do not call models, execute generated programs, or modify source result CSVs; no `.env` is needed. Run them from the repository root:

```bash
python stat/summarize_results.py
python stat/run_all_mcnemar.py
python stat/run_all_mcnemar.py --task code
python stat/finalize_stability_metrics.py
```

The scripts write the following summaries directly to `stat/`:

- `metrics_summary.csv`: overall and per-dataset QA metrics, plus code metrics.
- `mcnemar_all_models.csv` and `<model>_mcnemar_results.csv`: QA comparisons against CoT, CoVe, and DrHall.
- `mcnemar_all_models_code.csv` and `<model>_mcnemar_results_code.csv`: code comparisons against CoT and DrHall.
- `stability_results.csv` and `stability_summary.csv`: metrics for the original run and three repeats, with four-run means and sample standard deviations.

McNemar analysis uses paired outcomes, exact two-sided tests, Holm correction, and paired-bootstrap confidence intervals. QA comparisons can additionally include CoVe-SE using `--baselines CoT CoVe CoVe-SE DrHall`. Use `--help` for model selection and other options.

All rates are percentages:

- **RHR (lower is better):** incorrect final outputs / all outputs.
- **HRR (higher is better):** initially incorrect outputs repaired to correct / initially incorrect outputs.
- **OCR (lower is better):** initially correct outputs changed to incorrect / initially correct outputs.

QA label columns include `is_hallucination` and `recheck_hallucination*`; `YES`/`1` means incorrect and `NO`/`0` means correct. Code tables use `base_pass` and method-specific pass columns. Stability tables use `id,base,mutrepair`, optionally followed by summary columns. The statistics scripts recompute metrics from row-level labels rather than trusting embedded summaries.

**Empty labels count as incorrect**, including in paired tests. Unknown nonempty labels raise an error. A zero HRR or OCR denominator is reported as an empty value rather than zero.

## Run Experiments

The examples below use GPT-4o and are run from the repository root. Other models use the corresponding entry points and their own dataset filenames. Model calls incur provider charges, and stochastic outputs may differ from the included results.

Outputs are saved under the invoking model's `outputs/` directory. Several repair scripts resume from existing outputs. To keep the supplied results unchanged during a fresh run, use a separately named copy of the input CSV so that it produces a distinct output filename. Input paths are relative to the current working directory; default output paths are resolved from the model directory.

### Natural-Language QA

The provided QA inputs contain `Question`, `Answer`, and `base_response`. Repair uses the question and base response, not the reference answer.

```bash
python gpt-4o/repair_with_mutation.py --dataset_path gpt-4o/datasets/gpt-4o_dataset20251225_utf8_responses.csv --n_mutations 5
python gpt-4o/cot.py --dataset_path gpt-4o/datasets/gpt-4o_dataset20251225_utf8_responses.csv
python gpt-4o/drhall.py --dataset_path gpt-4o/datasets/gpt-4o_dataset20251225_utf8_responses.csv
```

The QA MutRepair output includes `mutation_list`, `answer_list`, and `final_answer_ra` (ranking), together with majority-voting (`final_answer_mv`) and confidence-score (`final_answer_cs`) alternatives and their cost fields. The main ranking result is written to `gpt-4o/outputs/gpt-4o_mutation_outputs_gpt-4o_dataset20251225_utf8_responses.csv`.

CoVe and CoVe-SE can be run as follows:

```bash
python gpt-4o/chain-of-verification-main/src/main.py --dataset_path gpt-4o/datasets/gpt-4o_dataset20251225_utf8_responses.csv
python gpt-4o/chain-of-verification-search-engine/src/main.py --dataset_path gpt-4o/datasets/gpt-4o_dataset20251225_utf8_responses.csv
```

Newly generated QA answers require correctness verification before statistical analysis. Generation does not automatically produce reviewed hallucination labels; any labels carried over from an input must not be treated as evaluations of new answers.

### Code Generation and Evaluation

LeetCode repair inputs require `task_id`, `problem_description`, `starter_code`, and `base_response`. Execution evaluation also requires `test` and `entry_point`, either in the repair output or in the original dataset supplied with `--dataset_path`.

```bash
python gpt-4o/repair_with_mutation_leetcode.py --dataset_path gpt-4o/datasets/gpt-4o_leetcode_responses.csv --n_mutations 5
python gpt-4o/cot_leetcode.py --dataset_path gpt-4o/datasets/gpt-4o_leetcode_responses.csv
python gpt-4o/drhall_leetcode.py --dataset_path gpt-4o/datasets/gpt-4o_leetcode_responses.csv
```

Repair outputs store the selected program in `final_answer`. To execute the MutRepair output against its associated tests:

```bash
python gpt-4o/auto_evaluation_leetcode.py --repair_paths gpt-4o/outputs/gpt-4o_mutrepair_gpt-4o_leetcode_responses.csv --method_names mutrepair --dataset_path gpt-4o/datasets/gpt-4o_leetcode_responses.csv --output_path gpt-4o/outputs/eval_code/evaluation_results.csv
```

The evaluator writes per-task pass/fail annotations and a companion `.metrics.json` file. It also accepts multiple `--repair_paths` with matching `--method_names`. Tests are used only for evaluation, not during MutRepair. Since evaluation executes generated Python code in subprocesses, use an isolated environment without sensitive files or credentials; a subprocess timeout is not a security sandbox.

### Generate Base Responses for New Inputs

Base responses are already included in the supplied datasets. To generate them for new CSVs:

```bash
python gpt-4o/tools/generate_response.py --dataset_path path/to/new_qa.csv
python gpt-4o/tools/generate_response_leetcode.py --dataset_path path/to/new_code.csv
```

The QA generator reads `Question`; the code generator reads `query` containing the complete programming prompt. Other columns are preserved. Generated files are saved in the model's `datasets/` directory as `<model>_<input-stem>_responses.csv`, with a `base_response` column.

### Full-Pipeline Stability Runs

The stability entry point repeats mutation generation, all candidate repairs, and ranking; it does not rerun ranking alone. It uses the full QA dataset by default:

```bash
python gpt-4o/repair_with_mutation_stability.py --runs 1 2 3 --workers 3 --dry-run
python gpt-4o/repair_with_mutation_stability.py --runs 1 2 3 --workers 3
```

The first command only displays the input and output paths. The second makes model calls. Runs are processed sequentially, with `--workers` controlling concurrency within each run. Answers and resumable caches are stored under `outputs/stability/`; preserve the caches when resuming with the same inputs and settings. New answers must be evaluated before they can be included in `eval_stability/` label tables and four-run summaries.

## Tests

```bash
python -m unittest discover -s test
```

These tests check configuration, shared entry points, label parsing, metric formulas, and paired statistics without running live model experiments.
