import pandas as pd
import ast

df = pd.read_csv("../triviaqa.csv", encoding="utf-8-sig")


def str_to_dict(s):
    try:
        return ast.literal_eval(s)
    except:
        return {}


df['Answer_dict'] = df['Answer'].apply(str_to_dict)


def dict_to_semicolon_answers(data_dict):
    if not isinstance(data_dict, dict):
        return ""

    fields_to_extract = [
        'Aliases',
        'HumanAnswers',
        'MatchedWikiEntityName',
        'NormalizedAliases',
        'NormalizedMatchedWikiEntityName',
        'NormalizedValue',
        'Value'
    ]

    all_answers = set()
    for field in fields_to_extract:
        value = data_dict.get(field)
        if isinstance(value, list):
            all_answers.update(value)
        elif isinstance(value, str) and value.strip():
            all_answers.add(value.strip())

    result = "; ".join(sorted(all_answers))
    return result


df['Answer_semicolon'] = df['Answer_dict'].apply(dict_to_semicolon_answers)

df.to_csv("triviaqa_semicolon.csv", index=False, encoding="utf-8-sig")
