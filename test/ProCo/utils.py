import time
import tiktoken
import g4f
import openai

encoding_name = "cl100k_base"
encoding = tiktoken.get_encoding(encoding_name)

SHOW = False
MAX_ITERATION = 1


def num_tokens_from_string(s):
    return len(encoding.encode(s))


def check_string(s):
    if s.strip() == "" or s.strip() == "IP access frequency is too high, please try again later":
        raise ValueError("Empty or invalid string encountered.")


def answer_by_bing_with_cost(prompt):
    start = time.perf_counter()
    response = g4f.ChatCompletion.create(
        provider=g4f.Provider.Bing,
        model=g4f.models.default,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    end = time.perf_counter()
    tokens = num_tokens_from_string(prompt) + num_tokens_from_string(response)
    check_string(response)
    return response, tokens, (end - start)


def answer_by_mixtral_with_cost(prompt):
    start = time.perf_counter()
    response = g4f.ChatCompletion.create(
        provider=g4f.Provider.PerplexityLabs,
        model=g4f.models.default,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    end = time.perf_counter()
    tokens = num_tokens_from_string(prompt) + num_tokens_from_string(response)
    check_string(response)
    return response, tokens, (end - start)


def answer_by_gpt_3_5_turbo_with_cost(prompt):
    start = time.perf_counter()
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    response = completion.choices[0].message.content
    end = time.perf_counter()
    tokens = num_tokens_from_string(prompt) + num_tokens_from_string(response)
    check_string(response)
    return response, tokens, (end - start)


def answer_by_model_key_with_cost(prompt, model_key):
    start = time.perf_counter()
    completion = openai.ChatCompletion.create(
        model=model_key,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    response = completion.choices[0].message.content
    end = time.perf_counter()
    tokens = num_tokens_from_string(prompt) + num_tokens_from_string(response)
    check_string(response)
    return response, tokens, (end - start)
