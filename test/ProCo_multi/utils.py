import random
import time
import tiktoken
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
encoding_name = "cl100k_base"
encoding = tiktoken.get_encoding(encoding_name)


def num_tokens_from_string(s):
    return len(encoding.encode(s))


def check_string(s):
    if s.strip() == "" or s.strip() == "IP access frequency is too high, please try again later":
        raise ValueError("Empty or invalid string encountered.")


def answer_by_model_key_with_cost(prompt, model_key, max_retries=3, base_delay=2.0):
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_key,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            response = completion.choices[0].message.content
            if not response or not response.strip():
                raise ValueError("Empty response from model")

            tokens = num_tokens_from_string(prompt) + num_tokens_from_string(response)
            check_string(response)

            return response, tokens

        except Exception as e:
            wait = base_delay * (attempt + 1) + random.uniform(0, 1)
            print(f"[Warning] Attempt {attempt + 1}/{max_retries} failed: {e}")
            print(f"Retrying after {wait:.1f}s ...")
            time.sleep(wait)

    print("[Error] Model repeatedly returned empty or invalid response.")
    return "ERROR: Empty or invalid model output", 0
