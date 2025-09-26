import json
import csv


def json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break

    if not isinstance(data, list):
        raise ValueError("JSON structure not supported: must be a list of objects")

    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    fieldnames = list(fieldnames)

    # Write CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Successfully converted {json_file} → {csv_file}")


json_to_csv("../verified-wikipedia-dev.json", "../triviaqa.csv")
