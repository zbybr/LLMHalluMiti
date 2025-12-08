SYSTEM_PROMPT = """The base response may contains hallucinations or factual errors. Verify facts step-by-step and produce 
one corrected, factual, one-sentence answer based on real-world truth. Do not include invented details or information 
from non-authoritative sources (e.g., advertisements, fan fiction, or marketing).

If the question explicitly asks about myths, legends, fiction, films, or other non-real contexts, answer within that 
fictional context but clearly label it as fictional and then you NEED to provide the accurate real‑world answer.

Only if the question is subjective, you can reply: "I have no idea."."""


MUTATION_PROMPT = """Given a question and a base response, create 5 different complete-sentence mutations of the 
response. Use varied rewriting strategies, such as replacing words with synonyms or antonyms, changing sentence 
structure, or introducing slight content changes by adding or removing a condition, altering viewpoint, omitting a 
detail, swapping a cause–effect relationship, or adding a small commonsense twist. If the base response is 
incomplete, rewrite it into a full sentence using the question’s context. Output all mutations as a numbered list 
without explanations."""

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

JUDGE_PROMPT = """Given a question and a list of answers from different reasoning paths, determine the final answer 
by majority voting. Identify answers with the same meaning, count their occurrences, and select the most frequent 
meaning. 'I have no idea.' is also a possible answer. If there are equal occurrences or the final selection can't be 
decided, you should think step by step, choose the most possible one.

Final answer(in exactly one sentence):
"""

VERBALIZED_SAMPLING_PROMPT = """You are a helpful assistant. You are given a question and a base response. The base 
response may contains hallucinations or factual errors. For each query, please generate a set of five possible 
correct answers for the original question, each within a separate <response> tag. Responses should each include a 
<text> and a numeric <probability>. Please sample at random from the [full distribution / tails of the distribution, 
such that the probability of each response is less than 0.10]."""