import pandas as pd
from scipy.stats import wilcoxon


def _load_and_unify(path: str, final_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if final_col != "final_hallucination":
        df = df.rename(columns={final_col: "final_hallucination"})
    # basic sanity checks
    required = {"Question", "is_hallucination", "final_hallucination"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def _merge_on_question(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    merged = df1.merge(df2, on="Question", suffixes=("_m1", "_m2"))
    if merged.empty:
        raise ValueError("No overlapping questions after merge. Check 'Question' alignment.")
    return merged


def _to01(series_yes_no: pd.Series) -> pd.Series:
    # YES -> 1, NO -> 0
    s = series_yes_no.astype(str).str.upper().str.strip()
    return (s == "YES").astype(int)


def wilcoxon_hrr(
    path1: str,
    path2: str,
    final_col1: str,
    final_col2: str,
    method1_name: str = "Method1",
    method2_name: str = "Method2",
    alternative: str = "two-sided",
):
    """
    HRR significance: compare methods on the subset where base responses are hallucinations.
    Data used for test: per-question post-repair hallucination indicator (1=still hallucinated, 0=correct).
    """
    df1 = _load_and_unify(path1, final_col1)
    df2 = _load_and_unify(path2, final_col2)

    # Subset: originally hallucinated
    df1_sub = df1[df1["is_hallucination"].astype(str).str.upper().str.strip() == "YES"].copy()
    df2_sub = df2[df2["is_hallucination"].astype(str).str.upper().str.strip() == "YES"].copy()

    merged = _merge_on_question(df1_sub, df2_sub)

    x = _to01(merged["final_hallucination_m1"])  # 1=still hallucination
    y = _to01(merged["final_hallucination_m2"])

    stat, p = wilcoxon(x, y, alternative=alternative, zero_method="wilcox")

    return {
        "metric": "HRR-test (on base-hallucinated subset)",
        "method1": method1_name,
        "method2": method2_name,
        "n_pairs": int(len(merged)),
        "statistic": float(stat),
        "p_value": float(p),
        "m1_corrected": int((x == 0).sum()),
        "m2_corrected": int((y == 0).sum()),
        "m1_still_hallu": int((x == 1).sum()),
        "m2_still_hallu": int((y == 1).sum()),
    }


def wilcoxon_rhr(
    path1: str,
    path2: str,
    final_col1: str,
    final_col2: str,
    method1_name: str = "Method1",
    method2_name: str = "Method2",
    alternative: str = "two-sided",
):
    """
    RHR significance: compare methods on ALL questions.
    Data used for test: per-question post-repair hallucination indicator (1=hallucinated, 0=non-hallucinated).
    """
    df1 = _load_and_unify(path1, final_col1)
    df2 = _load_and_unify(path2, final_col2)

    merged = _merge_on_question(df1, df2)

    x = _to01(merged["final_hallucination_m1"])
    y = _to01(merged["final_hallucination_m2"])

    stat, p = wilcoxon(x, y, alternative=alternative, zero_method="wilcox")

    return {
        "metric": "RHR-test (on full dataset)",
        "method1": method1_name,
        "method2": method2_name,
        "n_pairs": int(len(merged)),
        "statistic": float(stat),
        "p_value": float(p),
        "m1_hallu": int((x == 1).sum()),
        "m2_hallu": int((y == 1).sum()),
        "m1_non_hallu": int((x == 0).sum()),
        "m2_non_hallu": int((y == 0).sum()),
    }


def wilcoxon_ocr(
    path1: str,
    path2: str,
    final_col1: str,
    final_col2: str,
    method1_name: str = "Method1",
    method2_name: str = "Method2",
    alternative: str = "two-sided",
):
    """
    OCR significance: compare methods on the subset where base responses are originally correct (non-hallucinated).
    Data used for test: per-question over-correction indicator (1=became hallucinated, 0=remained non-hallucinated).
    """
    df1 = _load_and_unify(path1, final_col1)
    df2 = _load_and_unify(path2, final_col2)

    # Subset: originally correct (non-hallucination)
    df1_sub = df1[df1["is_hallucination"].astype(str).str.upper().str.strip() == "NO"].copy()
    df2_sub = df2[df2["is_hallucination"].astype(str).str.upper().str.strip() == "NO"].copy()

    merged = _merge_on_question(df1_sub, df2_sub)

    # over-corrected = final becomes hallucination
    x = _to01(merged["final_hallucination_m1"])  # 1=over-corrected
    y = _to01(merged["final_hallucination_m2"])

    stat, p = wilcoxon(x, y, alternative=alternative, zero_method="wilcox")

    return {
        "metric": "OCR-test (on base-correct subset)",
        "method1": method1_name,
        "method2": method2_name,
        "n_pairs": int(len(merged)),
        "statistic": float(stat),
        "p_value": float(p),
        "m1_over_corrected": int((x == 1).sum()),
        "m2_over_corrected": int((y == 1).sum()),
        "m1_remain_correct": int((x == 0).sum()),
        "m2_remain_correct": int((y == 0).sum()),
    }


def pretty_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return "<0.01"
    if p < 0.05:
        return "<0.05"
    return f"{p:.3f}"


if __name__ == "__main__":
    # Example usage (adapt paths/columns to your setup)
    path_name = "../ollama_outputs/qwen3_32b_mutation_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed_newlines.csv"
    path_cove = "../ollama_outputs/cot/qwen3_32b_cot_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed.csv"

    # Columns in your example
    final_col_name = "recheck_hallucination_ra"
    final_col_cove = "recheck_hallucination"

    res_hrr = wilcoxon_hrr(path_name, path_cove, final_col_name, final_col_cove,
                           method1_name="NAME", method2_name="CoVe-SE")
    res_rhr = wilcoxon_rhr(path_name, path_cove, final_col_name, final_col_cove,
                           method1_name="NAME", method2_name="CoVe-SE")
    res_ocr = wilcoxon_ocr(path_name, path_cove, final_col_name, final_col_cove,
                           method1_name="NAME", method2_name="CoVe-SE")

    for res in (res_hrr, res_rhr, res_ocr):
        print("=" * 80)
        print(res["metric"])
        print(f'{res["method1"]} vs {res["method2"]} | n={res["n_pairs"]}')
        print(f"stat={res['statistic']:.3f} | p={res['p_value']} ({pretty_p(res['p_value'])})")
        # print a couple key counts for sanity
        if "m1_corrected" in res:
            print(f"corrected: {res['method1']}={res['m1_corrected']}, {res['method2']}={res['m2_corrected']}")
        if "m1_over_corrected" in res:
            print(f"over-corrected: {res['method1']}={res['m1_over_corrected']}, {res['method2']}={res['m2_over_corrected']}")
