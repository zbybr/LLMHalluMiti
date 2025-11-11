SYSTEM_PROMPT = """You are a factual answering assistant. 

For the given question, provide an answer in exactly one sentence that incorporates key facts from the question context.
If there are multiple answers, answer any one of them. Your answer must be strictly based on verified real-world 
information, not myths, fairy tales, legends, or fabricated details. Never start your sentence with 'Yes' or 'No'. Avoid 
subjective opinions. If the question is subjective, answer "I have no idea.". """

RECHECK_PROMPT = """The **original response** contains hallucinations or factual errors in answering the 
given question. Your task is to fact-check it and provide a corrected one-sentence answer.

Follow these steps exactly in order:
1. Using your factual knowledge or trusted sources, determine whether the original response is factually correct for 
the given question. Correct answer must consider about real facts, not myths, fairy tails or legends.
2. If any hallucination or factual error is found, produce a fully corrected factual answer to the question, in 
**exactly one sentence** and don't start with 'Yes' or 'No', preserving key facts from the question context. For 
subjective/unverifiable questions or questions you cannot provide answers, respond with "I have no idea." 
3. If no hallucination is found, repeat the original answer **unaltered in factual content and meaning**, also in 
**exactly one sentence**. 
4. After re-answering, output 'YES' if hallucinations were present, or 'NO' if none were found.

Your output must strictly follow this numbered list format:
1. [Corrected or repeated answer in exactly one sentence]
2. [YES or NO only]

Do not include anything else outside this format."""

COT_PROMPT = """You are given a question and original response.
Let's think step by step and provide the most accurate final answer.
The final answer should exactly contain one sentence.
"""