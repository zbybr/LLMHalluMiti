import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def compare_methods_wilcoxon(
    path1,
    path2,
    method1_name,
    method2_name,
    final_col1="final_hallucination",
    final_col2="final_hallucination",
):
    """
    Compare two hallucination mitigation methods using Wilcoxon signed-rank test.

    Parameters:
    -----------
    path1 : str
        Path to first method's results CSV
    path2 : str
        Path to second method's results CSV
    method1_name : str
        Display name for first method
    method2_name : str
        Display name for second method
    final_col1 : str
        Column name for final_hallucination in first dataset (default: 'final_hallucination')
    final_col2 : str
        Column name for final_hallucination in second dataset (default: 'final_hallucination')

    Returns:
    --------
    dict : Dictionary containing test results and metrics
    """
    # Load datasets
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # Rename final_hallucination columns if needed
    if final_col1 != "final_hallucination":
        df1 = df1.rename(columns={final_col1: "final_hallucination"})
    if final_col2 != "final_hallucination":
        df2 = df2.rename(columns={final_col2: "final_hallucination"})

    # Filter to only rows where there was initially a hallucination
    df1_halu = df1[df1["is_hallucination"] == "YES"].copy()
    df2_halu = df2[df2["is_hallucination"] == "YES"].copy()

    print(f"Total base hallucinations in {method1_name}: {len(df1_halu)}")
    print(f"Total base hallucinations in {method2_name}: {len(df2_halu)}")

    # Merge on Question to align the datasets
    merged = df1_halu.merge(df2_halu, on="Question", suffixes=("_m1", "_m2"))

    print(f"\nCommon hallucinated questions: {len(merged)}")

    # Convert final_hallucination to numeric (YES=1, NO=0)
    method1_final = (merged["final_hallucination_m1"] == "YES").astype(int)
    method2_final = (merged["final_hallucination_m2"] == "YES").astype(int)

    # Perform Wilcoxon signed-rank test
    statistic, p_value = wilcoxon(method1_final, method2_final)

    # Calculate metrics
    m1_corrected = (method1_final == 0).sum()
    m2_corrected = (method2_final == 0).sum()
    m1_still_halu = (method1_final == 1).sum()
    m2_still_halu = (method2_final == 1).sum()

    # Print results
    print("\n" + "=" * 80)
    print(f"Wilcoxon Signed-Rank Test: {method1_name} vs {method2_name}")
    print("(Comparing methods on initially hallucinated responses)")
    print("=" * 80)
    print(f"\n{method1_name}:")
    print(
        f"  - Corrected (YES→NO): {m1_corrected} ({m1_corrected/len(merged)*100:.1f}%)"
    )
    print(
        f"  - Still hallucinating: {m1_still_halu} ({m1_still_halu/len(merged)*100:.1f}%)"
    )

    print(f"\n{method2_name}:")
    print(
        f"  - Corrected (YES→NO): {m2_corrected} ({m2_corrected/len(merged)*100:.1f}%)"
    )
    print(
        f"  - Still hallucinating: {m2_still_halu} ({m2_still_halu/len(merged)*100:.1f}%)"
    )

    print(f"\nWilcoxon Test Results:")
    print(f"  - Test Statistic: {statistic}")
    print(f"  - P-value: {p_value}")

    if p_value < 0.05:
        print(f"\n✓ Significant difference (p < 0.05)")
        if m1_corrected > m2_corrected:
            print(
                f"  {method1_name} corrected {m1_corrected - m2_corrected} more hallucinations than {method2_name}"
            )
        else:
            print(
                f"  {method2_name} corrected {m2_corrected - m1_corrected} more hallucinations than {method1_name}"
            )
    else:
        print(f"\n✗ No significant difference (p >= 0.05)")

    return {
        "method1": method1_name,
        "method2": method2_name,
        "p_value": p_value,
        "statistic": statistic,
        "method1_corrected": m1_corrected,
        "method2_corrected": m2_corrected,
        "total_compared": len(merged),
    }


if __name__ == "__main__":
    # Example usage
    results = compare_methods_wilcoxon(
        path1="gpt-5/outputs/gpt-5_mutation_outputs_gpt-5_dataset20251225_utf8_responses.csv",
        path2="gpt-5/outputs/cove-se/gpt-5_cove_se_outputs_gpt-5_dataset20251225_utf8_responses.csv",
        method1_name="NAME (Mutation)",
        method2_name="CoVe-SE",
        final_col1="recheck_hallucination_ra",
        final_col2="recheck_hallucination",
    )
