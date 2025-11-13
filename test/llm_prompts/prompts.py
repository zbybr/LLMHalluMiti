SYSTEM_PROMPT = """The **base response** contains hallucinations or factual errors in answering the given question. 
Think step by step, use your factual knowledge or trusted sources provide a corrected one-sentence answer based only on 
real facts. Do not include myths, fairy tails, bible or legends. Never begin your answer with 'Yes' or 'No'. If the question is 
subjective or cannot be answered factually, respond with "I have no idea."."""

MUTATION_PROMPT = """Given a question and a base response, generate 5 distinct mutations of the base response. Each 
mutation should be a complete, coherent sentence.

Think step by step, use diverse rewriting strategies to create mutations:
1. Change wording and structure: replace words with synonyms, shift between active/passive voice, reorder or combine 
sentences, or simplify and elaborate phrasing.
2. Introduce slight content changes: add or remove a condition, present an opposite viewpoint, omit a detail, 
swap a cause-and-effect relationship, or include a minor commonsense twist.
3. If the base response is not a complete sentence, expand or rewrite it using the question's context so that each
mutation is a grammatically complete and self-contained sentence.

Ensure each mutation reads naturally and remains relevant to the original question content. Do not label or explain the 
changes. Finally, output all mutations as a numbered list, with each item being one full sentence version."""

# RECHECK_PROMPT = """The **original response** contains hallucinations or factual errors in answering the
# given question. Your task is to fact-check it and provide a corrected one-sentence answer.
#
# Follow these steps exactly in order:
# 1. Using your factual knowledge or trusted sources, determine whether the original response is factually correct for
# the given question. Correct answer must consider real facts, not myths, fairy tails or legends.
# 2. If any hallucination or factual error is found, produce a fully corrected factual answer to the question, in
# **exactly one sentence** and don't start with 'Yes' or 'No', preserving key facts from the question context. For
# subjective/unverifiable questions or questions you cannot provide answers, respond with "I have no idea."
# 3. If no hallucination is found, repeat the original answer **unaltered in factual content and meaning**, also in
# **exactly one sentence**.
# 4. After re-answering, output 'YES' if hallucinations were present, or 'NO' if none were found.
#
# Your output must strictly follow this numbered list format:
# 1. [Corrected or repeated answer in exactly one sentence]
# 2. [YES or NO only]
#
# Do not include anything else outside this format."""

COT_PROMPT = """You are given a question and original response.
Let's think step by step and provide the most accurate final answer.
The final answer should exactly contain one sentence.
"""

JUDGE_PROMPT = """Given a list of answers from different reasoning paths, determine the final answer by majority voting. 
Identify answers with the same meaning, count their occurrences, and select the most frequent meaning. 
Output only the final consensus answer as one clear, complete sentence."""