import pandas as pd

input_path = "../datasets/truthfulqa1.3.csv"
output_path = "../datasets/truthfulqa1.3_combine.csv"

df = pd.read_csv(input_path, encoding="latin-1")

df["Answer"] = df["Best Answer"].fillna("") + ";" + df["Correct Answers"].fillna("")
df["Answer"] = df["Answer"].str.strip(";")

df_new = df[["Question", "Answer", "Incorrect Answers"]]

df_new.to_csv(output_path, index=False)


