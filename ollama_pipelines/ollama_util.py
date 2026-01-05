import random
import time

from langchain_ollama import ChatOllama


def safe_chat_call(messages, model_key, max_retries=20, base_delay=1.0):
    """
    Safe wrapper for ChatOllama with retries and detailed tracking.
    Returns: (content, token_cost)
    """
    llm = ChatOllama(
        model=model_key,
        base_url="http://localhost:11434",
        timeout=60,
        cache=False,
        num_predict=2048,
    )

    for attempt in range(max_retries):
        try:
            # Invoke the model
            response = llm.invoke(messages)
            content = response.content

            if not content or not content.strip():
                raise ValueError("Empty or null response from model")

            content = content.strip()

            print(response)
            # Get token usage from response metadata
            tokens = 0
            if hasattr(response, "response_metadata") and response.response_metadata:
                prompt_tokens = response.response_metadata.get("prompt_eval_count", 0)
                completion_tokens = response.response_metadata.get("eval_count", 0)
                tokens = prompt_tokens + completion_tokens

            return content, tokens

        except Exception as e:
            wait = base_delay * (attempt + 1) + random.uniform(0, 1)
            print(f"[Warning] Attempt {attempt+1}/{max_retries} failed: {e}")
            print(f"Retrying after {wait:.1f}s ...")
            time.sleep(wait)

    print("[Error] Model failed after multiple retries.")
    return "ERROR: Empty or invalid model output", 0
