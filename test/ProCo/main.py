import argparse
import os
import csv
from tqdm import tqdm
import prompt

parser = argparse.ArgumentParser(description="ProCo pipeline compare all variants")
parser.add_argument('--dataset_csv', required=True, help="Path to local dataset CSV")
parser.add_argument('--model_key', required=True, help="Model key for Advanced_ProCo")
args = parser.parse_args()

dataset_name = os.path.splitext(os.path.basename(args.dataset_csv))[0]
output_file = f"AllOutputs_{dataset_name}_Proco.csv"

Questions = []
Answers = []

with open(args.dataset_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        Questions.append(row['Question'])
        Answers.append(row['Answer'])

print(f"Loaded dataset '{dataset_name}' with {len(Questions)} questions.")

with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "question", "answer",  # 这里也保持和你的数据集一致
        "orig_final", "orig_token_cost", "orig_time_cost",
        "bing_final", "bing_token_cost", "bing_time_cost",
        "mixtral_final", "mixtral_token_cost", "mixtral_time_cost",
        "advanced_final", "advanced_token_cost", "advanced_time_cost"
    ])

    for idx, question in tqdm(list(enumerate(Questions)), total=len(Questions), desc="Processing"):
        orig_final, orig_tokens, orig_time = prompt.run_original_proco_pipeline(question)
        bing_final, bing_tokens, bing_time, _ = prompt.run_proco_pipeline(question, search_engine="bing")
        mix_final, mix_tokens, mix_time, _ = prompt.run_proco_pipeline(question, search_engine="mixtral")
        adv_final, adv_tokens, adv_time = prompt.run_advanced_proco_pipeline(question, args.model_key)

        writer.writerow([
            question, Answers[idx],  # 保持原文件的顺序
            orig_final, orig_tokens, orig_time,
            bing_final, bing_tokens, bing_time,
            mix_final, mix_tokens, mix_time,
            adv_final, adv_tokens, adv_time
        ])

print(f"Results saved to {output_file}.")
