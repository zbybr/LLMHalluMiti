import csv
import pandas as pd

df1 = pd.read_csv("output.csv", encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
df2 = pd.read_csv("gpt-4o_mutation_outputs_gpt-4o_dataset20251225_utf8_responses.csv", encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
m = df1.drop_duplicates("Question").set_index("Question")["final_answer_cs"]

df2["final_answer_cs"] = df2["Question"].map(m).combine_first(df2["final_answer_cs"])

df2.to_csv("final_output.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
