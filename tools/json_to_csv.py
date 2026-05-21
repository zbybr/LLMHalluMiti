import json
import csv


def json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break

    if not isinstance(data, list):
        raise ValueError("JSON structure not supported: must be a list of objects")

    _process_and_write_csv(data, json_file, csv_file)


def jsonl_to_csv(jsonl_file, csv_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if not data:
        raise ValueError("The JSONL file is empty or contains no valid JSON objects.")

    _process_and_write_csv(data, jsonl_file, csv_file)


def _process_and_write_csv(data, json_file, csv_file):
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    fieldnames = list(fieldnames)

    for item in data:
        for k, v in item.items():
            if isinstance(v, list):
                item[k] = '; '.join([str(x) for x in v])

    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Successfully converted {json_file} → {csv_file}")


if __name__ == "__main__":
    jsonl_to_csv("LeetCodeDataset-train.jsonl", "leetcode_train.csv")
