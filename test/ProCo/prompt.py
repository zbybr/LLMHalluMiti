import utils


# ---------------- Original ProCo ----------------
def run_original_proco_pipeline(question):
    """
    Original ProCo simplified version:
    - Retrieval using Bing by default
    - Reasoning with GPT-3.5
    - If incorrect, refine and re-answer once
    - Fixed single iteration
    """
    total_tokens = 0
    total_time = 0

    # Step 1: Retrieval (Bing)
    doc, doc_tokens, doc_time = utils.answer_by_bing_with_cost(question)
    total_tokens += doc_tokens
    total_time += doc_time

    # Step 2: Initial Answer
    refined_prompt = f"""Using the retrieved document:
{doc}
Answer the question accurately: "{question}"
Provide only the final answer."""
    ans, ans_tokens, ans_time = utils.answer_by_gpt_3_5_turbo_with_cost(refined_prompt)
    total_tokens += ans_tokens
    total_time += ans_time
    final_answer = ans

    # Step 3: Verification
    verification_prompt = f"Verify if the answer '{final_answer}' is factually correct for the question '{question}'. Reply with 'correct' or 'incorrect'."
    judgement, judge_tokens, judge_time = utils.answer_by_gpt_3_5_turbo_with_cost(verification_prompt)
    total_tokens += judge_tokens
    total_time += judge_time

    # Step 4: If incorrect, refine the question and re-answer once
    if "incorrect" in judgement.lower():
        refined_question = f"Refine the question: {question} avoiding incorrect info."
        ref_doc, ref_doc_tokens, ref_doc_time = utils.answer_by_bing_with_cost(refined_question)
        total_tokens += ref_doc_tokens
        total_time += ref_doc_time

        new_prompt = f"""Using the retrieved document:
{ref_doc}
Answer the question accurately: "{refined_question}"
Provide only the final answer."""
        new_ans, new_ans_tokens, new_ans_time = utils.answer_by_gpt_3_5_turbo_with_cost(new_prompt)
        total_tokens += new_ans_tokens
        total_time += new_ans_time
        final_answer = new_ans

    return final_answer, total_tokens, total_time


# ---------------- Engine-Separated ProCo: Bing / Mixtral ----------------
def run_proco_pipeline(question, search_engine="bing"):
    """
    Engine-separated ProCo:
    - search_engine: 'bing' or 'mixtral'
    - Retrieval via the specified engine
    - Reasoning and verification using GPT-3.5
    """
    total_tokens = 0
    total_time = 0

    # Step 1: Retrieval
    if search_engine == "bing":
        search_func = utils.answer_by_bing_with_cost
    elif search_engine == "mixtral":
        search_func = utils.answer_by_mixtral_with_cost
    else:
        raise ValueError("search_engine must be 'bing' or 'mixtral'")

    doc, doc_tokens, doc_time = search_func(question)
    total_tokens += doc_tokens
    total_time += doc_time

    # Step 2: Reasoning
    refined_prompt = f"""Using the retrieved document:
{doc}
Answer the question accurately: "{question}"
Provide only the final answer."""
    ans, ans_tokens, ans_time = utils.answer_by_gpt_3_5_turbo_with_cost(refined_prompt)
    total_tokens += ans_tokens
    total_time += ans_time
    final_answer = ans

    # Step 3: Verification
    verification_prompt = f"Verify if the answer '{final_answer}' is factually correct for the question '{question}'. Reply with 'correct' or 'incorrect'."
    judgement, judge_tokens, judge_time = utils.answer_by_gpt_3_5_turbo_with_cost(verification_prompt)
    total_tokens += judge_tokens
    total_time += judge_time

    return final_answer, total_tokens, total_time, None


# ---------------- Advanced ProCo (MODEL_KEY) ----------------
def run_advanced_proco_pipeline(question, model_key):
    """
    Advanced ProCo:
    - Uses the specified MODEL_KEY model
    - Does not rely on external search engines
    - The model itself performs search + reasoning + verification
    - Single iteration only
    """
    total_tokens = 0
    total_time = 0

    # Step 1: Model internal search + answer
    initial_prompt = f"""
You are an advanced AI model.
Search your internal knowledge or training data for relevant facts to answer the question.
Question: {question}
First, provide a short background context, then give your final answer clearly.
"""
    doc_and_answer, init_tokens, init_time = utils.answer_by_model_key_with_cost(initial_prompt, model_key)
    total_tokens += init_tokens
    total_time += init_time

    final_answer = doc_and_answer.strip()

    # Step 2: Verification
    verify_prompt = f"Verify if the answer '{final_answer}' is correct for the question '{question}'. Reply with 'correct' or 'incorrect'."
    judgement, judge_tokens, judge_time = utils.answer_by_model_key_with_cost(verify_prompt, model_key)
    total_tokens += judge_tokens
    total_time += judge_time

    # Step 3: If incorrect, refine and re-answer once
    if "incorrect" in judgement.lower():
        refine_prompt = f"""
You previously answered '{final_answer}' but that seems incorrect.
Please reconsider the question "{question}" and provide a corrected final answer.
"""
        refined_answer, ref_tokens, ref_time = utils.answer_by_model_key_with_cost(refine_prompt, model_key)
        total_tokens += ref_tokens
        total_time += ref_time
        final_answer = refined_answer.strip()

    return final_answer, total_tokens, total_time
