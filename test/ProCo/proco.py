import argparse
import os
import csv
from tqdm import tqdm
from pathlib import Path
import prompt


def run_pipeline(input_path, output_path):
    Questions = []
    Answers = []

    with open(input_path, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            Questions.append(row['Question'])
            Answers.append(row['Answer'])

    print(f"Loaded dataset '{input_path}' with {len(Questions)} questions.")

    with open(output_path, 'w', encoding='latin-1', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "question", "answer",
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
                question, Answers[idx],
                orig_final, orig_tokens, orig_time,
                bing_final, bing_tokens, bing_time,
                mix_final, mix_tokens, mix_time,
                adv_final, adv_tokens, adv_time
            ])

    print(f"Results saved to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProCo pipeline compare all variants")
    parser.add_argument('--dataset_path', required=True, help="Dataset path")
    parser.add_argument('--model_key', required=True, help="Model key for Advanced_ProCo")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()
    output_path = f"Proco_outputs_{dataset_name}.csv"

    run_pipeline(dataset_path, output_path)
