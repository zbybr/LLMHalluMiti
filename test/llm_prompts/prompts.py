SYSTEM_PROMPT = """For the question, please answer in 1 sentence including the question context. Please notice, 
your answer must consider about real facts, not myths, fairy tails or legends. If possible, do not include yes 
or no at the beginning of the sentence."""

RECHECK_PROMPT = """You are given a question and a response, this response contain hallucinations. Please 
carefully recheck and provide: 
1. A corrected or verified response, please answer in 1 sentence including the question context. 
2. A short statement whether the original was hallucinated or not. You should answer YES or NO first.
Please notice just return the numbered list. Do not add anything else."""

RECHECK_PROMPT_NOSPLIT = """The above response from you contain hallucinations. Please carefully recheck and provide: 
1. A corrected or verified response, please answer in 1 sentence including the question context. 
2. A short statement whether the original was hallucinated or not. You should answer YES or NO first.
Please notice just return the numbered list. Do not add anything else."""
