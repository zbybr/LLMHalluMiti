import pandas as pd

def extract_name_success_cove_fail(
    name_csv: str,
    cove_csv: str,
    out_csv: str = "NAME_success_CoVe_fail_cases.csv",
    key_col: str = "Question",
    base_halu_col: str = "is_hallucination",
    base_resp_col: str = "base_response",
    name_final_col: str = "recheck_hallucination_ra",   # NAME 的最终是否幻觉列：YES/NO
    cove_final_col: str = "recheck_hallucination",      # CoVe 的最终是否幻觉列：YES/NO
):
    # Load
    df_name = pd.read_csv(name_csv, encoding="utf-8-sig")
    df_cove = pd.read_csv(cove_csv, encoding="utf-8-sig")

    # Keep only what we need from CoVe
    df_cove = df_cove[[key_col, cove_final_col]].copy()

    # Merge (base_response + base hallucination label come from NAME)
    merged = df_name.merge(df_cove, on=key_col, how="inner", suffixes=("", "_COVE"))

    # Filter: base hallucination + NAME fixed + CoVe failed
    wins = merged[
        (merged[base_halu_col] == "YES") &
        (merged[name_final_col] == "NO") &
        (merged[cove_final_col] == "YES")
    ].copy()

    # Output columns (keep clean, easy for case study)
    out_cols = [
        key_col,
        base_resp_col if base_resp_col in wins.columns else None,
        base_halu_col,
        name_final_col,
        cove_final_col,
    ]
    out_cols = [c for c in out_cols if c is not None]

    # Add some helpful answer/output columns if present
    extra_cols = [c for c in wins.columns if any(k in c.lower() for k in ["final", "answer", "response", "output"])]
    for c in extra_cols:
        if c not in out_cols:
            out_cols.append(c)

    wins[out_cols].to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("Extracted cases: base hallucination + NAME fixed + CoVe failed")
    print("=" * 70)
    print("Merged rows:", len(merged))
    print("Selected rows:", len(wins))
    print("Saved:", out_csv)

    return wins[out_cols]


if __name__ == "__main__":
    extract_name_success_cove_fail(
        name_csv="../gpt-4o/outputs/gpt-4o_mutation_outputs_gpt-4o_dataset20251225_utf8_responses.csv",
        cove_csv="../gpt-4o/outputs/cove/gpt-4o_cove_outputs_gpt-4o_dataset20251225_utf8_responses.csv",
        out_csv="NAME_success_CoVe_fail_cases.csv",
        key_col="Question",
        base_halu_col="is_hallucination",
        base_resp_col="base_response",
        name_final_col="recheck_hallucination_ra",
        cove_final_col="recheck_hallucination",
    )
