import pandas as pd
import random
df = pd.read_csv("gpt-4o_mutation_outputs_gpt-4o_dataset20251225_utf8_responses.csv")
def parse_answers(text):
    return [x.strip() for x in str(text).split("\n") if x.strip()]

random.seed(42)
final_answers = []
for _, row in df.iterrows():
    answers = parse_answers(row["answer_list"])
    if len(answers) == 0:
        final_answers.append("")
    else:
        final_answers.append(random.choice(answers))

df["final_answer_nr"] = final_answers
df.to_csv("gpt-4o_mutation_outputs_gpt-4o_dataset20251225_utf8_responses_nr.csv", index=False)
