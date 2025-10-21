import time
import tiktoken
# import g4f
# from g4f.client import Client
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


def answer_by_model_key_with_cost(prompt, model_key):
    start = time.perf_counter()
    completion = client.chat.completions.create(
        model=model_key,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    response = completion.choices[0].message.content
    end = time.perf_counter()
    tokens = num_tokens_from_string(prompt) + num_tokens_from_string(response)
    check_string(response)
    return response, tokens, (end - start)
