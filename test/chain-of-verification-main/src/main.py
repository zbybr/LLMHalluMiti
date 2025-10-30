import argparse
from dotenv import load_dotenv
from pprint import pprint
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from route_chain import RouteCOVEChain
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '..', '.env')
load_dotenv(dotenv_path=ENV_PATH, override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


def process_question(question, base_response, model_name, temperature, max_tokens, show_steps):
    chain_llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    route_llm = ChatOpenAI(model_name=model_name, temperature=0.1, max_tokens=1024)

    router_cove_chain_instance = RouteCOVEChain(question, route_llm, chain_llm, show_steps)
    router_cove_chain = router_cove_chain_instance()
    result = router_cove_chain({
        "original_question": question,
        "baseline_response": base_response
    })

    if show_steps:
        print("\n" + 80 * "#" + "\n")
        pprint(result)

    print("\n" + 80 * "#" + "\n")
    print(f"Final Answer: {result['final_answer']}")
    return result["final_answer"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chain of Verification (CoVE) parser.')
    parser.add_argument('--question', type=str, required=False, help='Single question to ask')
    parser.add_argument('--base_response', type=str, required=False, help='Base response to verify')
    parser.add_argument('--dataset_path', type=str, required=False, help='Dataset path')
    parser.add_argument('--model_key', type=str, required=False, default="gpt-4o", help='Model key')
    parser.add_argument('--temperature', type=float, required=False, default=0.1, help='LLM temperature')
    parser.add_argument('--max_tokens', type=int, required=False, default=2048, help='Maximum tokens')
    parser.add_argument('--show_intermediate_steps', type=bool, required=False, default=True,
                        help='Show intermediate steps')

    args = parser.parse_args()

    if args.dataset_path:
        dataset_path = args.dataset_path
        df = pd.read_csv(dataset_path)
        print(f"Processing {len(df)} questions from dataset: {args.dataset}")
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing QA"):
            start = time.time()
            question = row["Question"]
            base_response = row['base_response']
            final_answer = process_question(question, base_response, args.model_key, args.temperature, args.max_tokens, args.show_intermediate_steps)
            end = time.time()
            print("===================================")
            print(f"Question: {question}")
            print(f"Base Response: {base_response}")
            print(f"Final Answer: {final_answer} (tokens={tokens}, time={end - start:.4f}s)")

            # Save results into dataframe
            df.loc[index, "final_answer"] = final_answer
            df.loc[index, "token_cost"] = tokens
            df.loc[index, "time_cost"] = end - start

        dataset_name = str(Path(dataset_path).stem).lower()
        output_path = f"{args.model_key}_outputs_{dataset_path}.csv"
        df.to_csv(output_path, index=False)
        print(f"Output saved at {output_path}")

    elif args.question:
        process_question(args.question, args.base_response, args.model_key, args.temperature, args.max_tokens,
                         args.show_intermediate_steps)
    else:
        print("Please provide either --question or --dataset")
