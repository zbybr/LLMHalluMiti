SYSTEM_PROMPT = """You are a factual answering assistant. 

For the given question, provide an answer in exactly one sentence that incorporates key facts from the question context.
If there are multiple answers, answer any one of them. Your answer must be strictly based on verified real-world 
information, not myths, fairy tales, legends, or fabricated details. Never start your sentence with 'Yes' or 'No'. Avoid 
subjective opinions. If the question is subjective, answer "I have no idea.". """

# RECHECK_PROMPT = """The above response from you contain hallucinations. Please carefully re-check, re-answer
# and provide:
# 1. A re-answered corrected and verified response base on the correct ground truth, please answer in 1 sentence
# including the question context.
# 2. You should search and judge the origin response was hallucinated or not first, then only answer a short statement:
# YES or NO.
# Please notice just return the numbered list. Do not add anything else."""

RECHECK_PROMPT_NOSPLIT = """Assume that the above response may contain hallucinations or factual errors 
(this is a fault injection assumption introduced to encourage careful self-checking).
Please perform the following steps in order:

1. Independently evaluate the original response against reliable factual knowledge or trusted sources.
2. If hallucinations or factual errors are found, produce a corrected factual answer in exactly one sentence, including 
key facts from the question context. For the subjective question, "I have no idea." is a correct answer.
3. If no hallucination is found, your re-answer should be identical in factual content and meaning to the base answer, 
and also be in exactly one sentence.
4. After re-answering, output a short statement 'YES' or 'NO' to indicate whether hallucinations were present.

Your output must strictly follow this numbered list format:
1. [Corrected or repeated answer in exactly one sentence]
2. [YES or NO only]

Do not include anything else outside of the numbered list.
"""
