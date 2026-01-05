import argparse
import csv
import os
import time
from pathlib import Path
from pprint import pprint

import pandas as pd
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_ollama import ChatOllama
from route_chain import RouteCOVEChain
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "..", ".env")


def process_question(
    question, base_response, model_name, temperature, max_tokens, show_steps
):
    chain_llm = ChatOllama(
        model=model_name, temperature=temperature, num_predict=max_tokens
    )
    route_llm = ChatOllama(model=model_name, temperature=0.1, num_predict=2048)
    router_cove_chain_instance = RouteCOVEChain(
        question, route_llm, chain_llm, show_steps
    )
    router_cove_chain = router_cove_chain_instance()
    result = router_cove_chain.invoke(
        {"original_question": question, "baseline_response": base_response}
    )

    if show_steps:
        print("\n" + 80 * "#" + "\n")
        pprint(result)

    print("\n" + 80 * "#" + "\n")
    print(f"Final Answer: {result['final_answer']}")
    return result["final_answer"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chain of Verification (CoVE) pipeline with cost tracking."
    )
    parser.add_argument(
        "--question", type=str, required=False, help="Single question to ask"
    )
    parser.add_argument(
        "--base_response", type=str, required=False, help="Base response to verify"
    )
    parser.add_argument("--dataset_path", type=str, required=False, help="Dataset path")
    parser.add_argument(
        "--model_key", type=str, required=False, default="gpt-4o", help="Model key"
    )
    parser.add_argument(
        "--temperature", type=float, required=False, default=0.1, help="LLM temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, required=False, default=2048, help="Maximum tokens"
    )
    parser.add_argument(
        "--show_intermediate_steps",
        type=bool,
        required=False,
        default=True,
        help="Show intermediate steps",
    )

    args = parser.parse_args()
    if args.dataset_path:
        dataset_path = args.dataset_path
        dataset_name = str(Path(dataset_path).stem).lower()
        output_path = (
            f"../../outputs/cove-se/{args.model_key}_cove_se_outputs_{dataset_name}.csv"
        )
        df = pd.read_csv(dataset_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

        if os.path.exists(output_path):
            print(f"Resuming from existing output file: {output_path}")
            df_out = pd.read_csv(
                output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
            )
            df = df.merge(
                df_out[["Question", "final_answer", "token_cost", "time_cost"]],
                on="Question",
                how="left",
                suffixes=("", "_saved"),
            )
        else:
            df["final_answer"] = ""
            df["token_cost"] = None
            df["time_cost"] = None
        df_todo = df[
            df["final_answer"].isna()
            | (df["final_answer"].astype(str).str.strip() == "")
        ]

        print(f"Total questions: {len(df)}")
        print(f"Already processed: {len(df) - len(df_todo)}")
        print(f"Remaining to process: {len(df_todo)}")

        print(f"Processing {len(df_todo)} questions from dataset: {args.dataset_path}")
        for index, row in tqdm(
            df_todo.iterrows(), total=len(df_todo), desc="Processing QA"
        ):
            max_retries = 20
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    with get_openai_callback() as cb:
                        start = time.time()
                        question = row["Question"]
                        base_response = row["base_response"]
                        final_answer = process_question(
                            question,
                            base_response,
                            args.model_key,
                            args.temperature,
                            args.max_tokens,
                            args.show_intermediate_steps,
                        )
                        end = time.time()
                        tokens = cb.total_tokens
                        print("===================================")
                        print(f"Question: {question}")
                        print(f"Base Response: {base_response}")
                        print(
                            f"Final Answer: {final_answer} (tokens={tokens}, time={end - start:.4f}s)"
                        )

                        # Save results into dataframe
                        df.loc[df["Question"] == question, "final_answer"] = (
                            final_answer
                        )
                        df.loc[df["Question"] == question, "token_cost"] = tokens
                        df.loc[df["Question"] == question, "time_cost"] = end - start
                        success = True
                except Exception as e:
                    retry_count += 1
                    print(f"Error on attempt {retry_count}/{max_retries}: {str(e)}")
                    if retry_count < max_retries:
                        print(f"Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        print(
                            f"Failed after {max_retries} attempts. Skipping question."
                        )
                        df.loc[df["Question"] == question, "final_answer"] = "ERROR"
                        df.loc[df["Question"] == question, "token_cost"] = 0
                        df.loc[df["Question"] == question, "time_cost"] = 0

            df.to_csv(output_path, encoding="utf-8-sig", index=False, quoting=csv.QUOTE_ALL)
        print(f"Output saved at {output_path}")

    elif args.question:
        process_question(
            args.question,
            args.base_response,
            args.model_key,
            args.temperature,
            args.max_tokens,
            args.show_intermediate_steps,
        )
    else:
        print("Please provide either --question or --dataset")
