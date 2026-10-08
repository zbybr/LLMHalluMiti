"""Published result paths adapted from the original all-model statistics script."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt-4o", "gpt-5", "gemini", "qwen3")

MODEL_FILES = {
    "gpt-4o": {
        "MutRepair": ROOT / "gpt-4o/outputs/gpt-4o_mutation_outputs_gpt-4o_dataset20251225_utf8_responses.csv",
        "CoT": ROOT / "gpt-4o/outputs/cot/gpt-4o_cot_outputs_gpt-4o_dataset20251225_utf8_responses.csv",
        "CoVe": ROOT / "gpt-4o/outputs/cove/gpt-4o_cove_outputs_gpt-4o_dataset20251225_utf8_responses.csv",
        "DrHall": ROOT / "gpt-4o/outputs/drhall/gpt-4o_drhall_ecmr3_gpt-4o_dataset20251225_utf8_responses.csv",
    },
    "gpt-5": {
        "MutRepair": ROOT / "gpt-5/outputs/gpt-5_mutation_outputs_gpt-5_dataset20251225_utf8_responses.csv",
        "CoT": ROOT / "gpt-5/outputs/cot/gpt-5_cot_outputs_gpt-5_dataset20251225_utf8_responses.csv",
        "CoVe": ROOT / "gpt-5/outputs/cove/gpt-5_cove_outputs_gpt-5_dataset20251225_utf8_responses.csv",
        "DrHall": ROOT / "gpt-5/outputs/drhall/gpt-5_drhall_ecmr3_gpt-5_dataset20251225_utf8_responses.csv",
    },
    "gemini": {
        "MutRepair": ROOT / "gemini/outputs/gemini_mutation_outputs_gemini_dataset20251225_utf8_responses.csv",
        "CoT": ROOT / "gemini/outputs/cot/gemini_cot_outputs_gemini_dataset20251225_utf8_responses.csv",
        "CoVe": ROOT / "gemini/outputs/cove/gemini_cove_outputs_gemini_dataset20251225_utf8_responses.csv",
        "DrHall": ROOT / "gemini/outputs/drhall/gemini_drhall_ecmr3_gemini_dataset20251225_utf8_responses.csv",
    },
    "qwen3": {
        "MutRepair": ROOT / "qwen3/outputs/qwen3_32b_mutation_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed_newlines.csv",
        "CoT": ROOT / "qwen3/outputs/cot/qwen3_32b_cot_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed.csv",
        "CoVe": ROOT / "qwen3/outputs/cove/qwen3_32b_cove_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed.csv",
        "DrHall": ROOT / "qwen3/outputs/drhall/qwen3-32b_drhall_ecmr3_qwen3_32b_dataset20251225_utf8_sig_responses_fixed.csv",
    },
}


def qa_files(model: str) -> dict[str, Path]:
    files = dict(MODEL_FILES[model])
    files["CoVe-SE"] = files["CoVe"].parent.parent / "cove-se" / files["CoVe"].name.replace("_cove_", "_cove_se_")
    return files


def code_files(model: str) -> dict[str, Path]:
    directory = ROOT / model / "outputs" / "eval_code"
    return {method: directory / f"{method.lower()}.csv" for method in ("MutRepair", "CoT", "DrHall")}

