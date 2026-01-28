import pandas as pd
import csv

df = pd.read_csv("qwen3_32b_cove_se_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed.csv", encoding="latin-1", quoting=csv.QUOTE_ALL)
df.to_csv("qwen3_32b_cove_se_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)