import pandas as pd

df = pd.read_csv("hotpot_test_fullwiki_v2.csv", encoding="iso-8859-1")

df["answer"] = df[["answer1", "answer2"]].apply(
    lambda row: "; ".join(row.dropna().astype(str).str.strip()), axis=1
)

df_result = df[["question", "answer"]]

df_result.to_csv("merged_answers.csv", index=False, encoding="iso-8859-1")