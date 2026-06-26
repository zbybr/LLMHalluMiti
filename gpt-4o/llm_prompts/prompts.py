SYSTEM_PROMPT = """The base response may contains hallucinations or factual errors. Verify facts step-by-step and 
produce one corrected, factual, one-sentence answer based on real-world truth. Do not include invented details or 
information from non-authoritative sources (e.g., advertisements, fan fiction, or marketing).

If the question explicitly asks about myths, legends, fiction, films, or other non-real contexts, answer within that 
fictional context but clearly label it as fictional and then you NEED to provide the accurate real‑world answer.

Only if the question is subjective, you can reply: "I have no idea.".

Return the final answer sentence, the final answer should exactly contain one sentence.
"""

MUTATION_PROMPT = """\
You are an expert at reformulating answer sentences.

Given a question and its base response, generate {n} diverse mutations of the \
response. Each mutation must be a single, grammatically complete sentence that \
directly answers the question. If the base response is incomplete or fragmentary, \
first rewrite it into a full sentence using the question's context, then mutate.

Apply the following metamorphic relation types (use each at least once):

1. Meaning-Preserving Rewrite
   Restate the same fact with different surface form: replace words with \
synonyms; reorder modifiers; change phrasing or word choice. The asserted \
fact must remain identical.

2. Structural Transformation
   Change sentence structure without altering the claim: switch active ↔ \
passive voice; swap subject and object with an adapted verb; convert between \
a statement and a clausal/appositive construction. The asserted fact must \
remain identical.

3. Semantic Polarity Shift
   Introduce a targeted semantic change that may expose a hidden assumption: \
replace the predicate with an antonym; negate the assertion; add or remove a \
qualifying condition; alter the viewpoint, a cause–effect link, or a single \
detail.

RULES:
- Do NOT verify or fix the base response. Mutate it faithfully even if it is \
factually wrong — preserving any potential error is required.
- Each mutation must stand alone and remain answerable as a response to the \
question.
- Keep each mutation to one self-contained sentence.
- Output all mutations as a numbered list and nothing else — no explanations, \
no labels, no commentary.

Output exactly {n} numbered lines:
1. <mutation 1>
2. <mutation 2>
...
"""

COT_PROMPT = """You are given a question and original response.
Let's think step by step and provide the most accurate final answer.
The final answer should exactly contain one sentence.
"""

VOTING_PROMPT = """Given a question and a list of answers from different reasoning paths, determine the final answer 
by majority voting. Identify answers with the same meaning, count their occurrences, and select the most frequent 
meaning. 'I have no idea.' is also a possible answer. If there are equal occurrences or the final selection can't be 
decided, you should think step by step, choose the most possible one.

Final answer should be in exactly one sentence.
"""

CONFIDENCE_SCORE_PROMPT = """You are a helpful assistant. I have a question and six potential answer sentences. For 
each answer sentence, please evaluate its relevance and quality in addressing the question. Provide a numeric number 
representing the model's confidence score for the factual correctness of the answer (range 0.00–1.00, two decimal places).

Instructions:
1. Fictional Context Rule: If the question asks about myths, legends, fiction, films, or other non-real contexts, answer 
within the fictional context, but clearly label it as fictional, and then provide the real-world answer.
2. Subjective Question Rule: If the question is subjective or opinion-based, and you cannot determine an objective answer, 
respond with "I have no idea".
3. The sentence with the highest confidence score should be identified as the best answer to the original question.

Final Step:
The best answer is the one with the highest confidence score. Only return this highest score answer sentence.
"""

RANKING_PROMPT = """You are a helpful assistant. I have a question and six potential answer sentences. I would like you 
to compare each answer against the others and rank them from best to worst based on the ground truth criteria in 
addressing the question that I will provide. Each answer should be compared to every other answer, and you should assign 
a ranking to each comparison.

Ground Truth Criteria for Ranking:
1. If the question explicitly asks about myths, legends, fiction, films, or other non-real contexts: Answer within the 
fictional context, but clearly label it as fictional. After that, you must provide the accurate real-world answer (e.g., 
real-world facts, history, or scientific information).
2. If the question is subjective (i.e., based on opinion, interpretation, or preference): If unsure, you may respond 
with "I have no idea."
3. For all other questions: Rank the answers based on their relevance, clarity, and accuracy in addressing the original 
question.

Instructions:
1. Please create a 6x6 matrix where each row represents an answer sentence, and each column represents a comparison of 
that answer sentence with the others.
2. he matrix should contain numbers from 1 to 6, where 1 is the best rank (highest quality), and 6 is the worst rank (
lowest quality).
3. In each cell, compare the corresponding row's answer to the sentence listed in the column. The lower the number, the 
better the answer.
4. Use the ground truth criteria to guide your rankings.
5. The highest-ranked answer (i.e., the one with the lowest total ranking) is considered the best answer to the original 
question.

Final Step:
The best answer is the one with the lowest rank in the matrix. Only return this highest ranking answer sentence.
"""

REFINE_PROMPT = """You are may given a paragraph, you need to find the final answer sentence in this paragraph. And ONLY 
return this answer sentence, DO NOT output the ranking or confidence score."""

LLM_JUDGE_PROMPT = """You are given a correct answer and another context, your task is to judge the final answer of the 
context is correct or not according to the given correct answer. Only return YES or NO."""

COT_PROMPT_LEETCODE = """You are an expert Python programmer. You will be given a LeetCode problem specification, 
starter code, and a draft solution.
Think step-by-step internally to evaluate the draft solution for correctness, edge cases, and efficiency. 
CRITICAL REQUIREMENT: Your final response MUST contain ONLY the executable Python code block. Do NOT include any 
explanations, introduction, markdown text outside the code block, or commentary. 
"""

MUTATION_LEETCODE_PROMPT = """\
You are an expert Python engineer specialising in code refactoring.

Given a LeetCode problem and a base solution, generate {n} diverse mutations of \
the solution. Each mutation must be a COMPLETE, RUNNABLE Python solution that \
preserves the class/function signature shown in the starter code.

Apply the following metamorphic relation types (use each at least once):

1. Meaning-Preserving Rewrite
   Rename local variables or helper functions to semantically similar names; \
convert a list comprehension to an equivalent for-loop or vice versa; reformat \
multi-line expressions; swap equivalent built-ins (e.g. `len(x) == 0` → `not x`). \
The logic and algorithm must remain identical.

2. Structural Transformation
   Change control-flow structure without altering the algorithm: \
for-loop ↔ while-loop; recursion ↔ iteration; extract a repeated block into a \
helper function or inline an existing helper; reorder independent statements.

3. Semantic Polarity Shift
   Introduce a targeted semantic inversion that may expose hidden assumptions: \
negate a boolean condition (`if x` → `if not x`); swap a comparison operator \
(`<` ↔ `>`; `<=` ↔ `>=`); alter a boundary value by ±1 \
(e.g. `n - 1` → `n`, `range(n)` → `range(n + 1)`).

4. Algorithm / Data-Structure Variant
   Replace the core algorithm or data structure with a plausible alternative: \
BFS ↔ DFS; two-pointer ↔ sliding window; dict ↔ sorted list; \
sort-then-scan ↔ heap; memoisation ↔ tabulation.

RULES:
- Do NOT fix bugs. Mutate faithfully even if the base code is wrong.
- Do NOT add any explanation or prose outside the code blocks.
- Every mutation must include all necessary imports.

Output exactly {n} numbered blocks and nothing else:
1.
```python
<mutation 1>
```
2.
```python
<mutation 2>
```
"""

REPAIR_LEETCODE_PROMPT = """\
You are an expert Python engineer performing a critical code review.

The code below is suspected to contain one or more of the following faults:
  - Hallucinated API call: a method, function, or module that does not exist in \
Python's standard library or common third-party packages.
  - Logic error: incorrect algorithm, wrong operator, or misplaced condition.
  - Boundary / off-by-one error: incorrect index, loop range, or comparison that \
fails on edge cases (empty input, single element, maximum value).
  - Type or return-value mismatch: incompatible operands, wrong return type, or \
missing return statement.

Instructions:
1. Trace the code line by line against the problem description and starter code.
2. Identify which fault category (if any) is present.
3. If the code is already correct, return it UNCHANGED.
4. If you find a fault, produce a fully corrected solution.

Return ONLY a single fenced Python code block — no explanation, no diff, no prose.

```python
<corrected solution>
```
"""

PAIRWISE_JUDGE_LEETCODE_PROMPT = """\
You are an impartial judge evaluating two Python solutions for a LeetCode problem.

Scoring criteria (in priority order):
1. Functional correctness: does the solution correctly handle all cases described \
in the problem, including edge cases?
2. Absence of hallucinated APIs: does the code avoid calling non-existent \
functions, methods, or modules?
3. Algorithmic soundness: is the algorithm logically correct and efficient?
4. Code quality: is the code readable, idiomatic, and free of dead code?

Assign candidate A a score in [1, {n}].
1 = candidate A is the best possible; {n} = candidate A is the worst.
Compare relative to candidate B: if A is better, give A a low score; \
if B is better, give A a high score.

Respond with ONLY a single integer — nothing else.
"""

PARAPHRASE_PROMPT_DRHALL_QA = """\
You are an expert at reformulating questions.

Given a question, generate {k} diverse paraphrases that preserve the EXACT
same meaning and ask for the EXACT same information, but vary the wording
and/or sentence structure. Apply the following strategies in order
(cycle through if k > 3):

1. Word-level substitution
   Replace nouns, verbs, and adjectives with synonyms while keeping the
   sentence structure as close to the original as possible.

2. Structure-level substitution
   Reorder clauses, change active/passive voice, or convert between a direct
   question and an imperative request, while keeping the vocabulary intact.

3. Combined substitution
   Apply both word-level and structure-level changes simultaneously.

Rules:
- Every paraphrase must ask for exactly the same answer as the original.
- Do NOT narrow, broaden, or shift the question's intent.
- Do NOT answer the question.
- Output exactly {k} numbered paraphrases and nothing else.

Format:
1. <paraphrase 1>

2. <paraphrase 2>
"""

REPAIR_PROMPT_DRHALL_QA = """\
You are a knowledgeable and careful assistant reviewing a candidate answer.

Carefully check whether the candidate answer correctly and completely answers
the question below, based on real-world truth.

- If the answer is correct, return it UNCHANGED.
- If it contains any factual error, hallucinated detail, or omission, return a
  corrected, factual answer based on real-world truth. Do not include invented
  details or information from non-authoritative sources (e.g. advertisements,
  fan fiction, or marketing).

If the question explicitly asks about myths, legends, fiction, films, or other
non-real contexts, answer within that fictional context but clearly label it as
fictional and then provide the accurate real-world answer.

Only if the question is subjective, you may reply: "I have no idea."

Return ONLY the final answer sentence. The final answer should contain exactly
one sentence — no preamble, no explanation, no meta-commentary.
"""

CONSISTENCY_VOTE_PROMPT_DRHALL_QA = """\
You are an impartial judge measuring agreement among candidate answers to a
question.

You are given a question and a list of candidate answers from different
reasoning paths. Group the answers by their CORE claim: two answers belong to
the same group if and only if they assert the same essential fact(s) in response
to the question, even if phrased differently. Ignore differences in wording,
length, or elaboration. 'I have no idea.' is also a possible answer.

Identify the LARGEST group (the answer most candidates agree on). If several
groups tie for largest, think step by step and choose the most factually
supported one.

Return ONLY one answer sentence that best represents that largest group. The
final answer should contain exactly one sentence — nothing else.
"""

PARAPHRASE_PROMPT_DRHALL = """\
You are an expert at reformulating technical problem descriptions.

Given a LeetCode problem description, generate {k} diverse paraphrases that
preserve the EXACT same requirements and constraints, but vary the wording
and/or sentence structure.  Apply the following strategies in order
(cycle through if k > 3):

1. Word-level substitution
   Replace nouns, verbs, and adjectives with synonyms while keeping the
   sentence structure as close to the original as possible.

2. Structure-level substitution
   Reorder clauses, split compound sentences, or change active/passive voice
   while keeping the original vocabulary as intact as possible.

3. Combined substitution
   Apply both word-level and structure-level changes simultaneously.

Rules:
- Every paraphrase must be a complete, self-contained problem description.
- Do NOT simplify, add, or remove any constraints or requirements.
- Output exactly {k} numbered paraphrases and nothing else.

Format:
1. <paraphrase 1>

2. <paraphrase 2>
"""

REPAIR_PROMPT_DRHALL = """\
You are an expert Python programmer reviewing a candidate solution to a \
LeetCode problem.

Carefully check whether the candidate solution correctly and completely \
solves the problem described below.

- If the solution is correct, return it UNCHANGED.
- If it contains any error (wrong logic, non-existent API, boundary mistake, \
wrong return value, missing edge case), return a corrected solution that \
matches the starter code signature exactly.

Return ONLY a single fenced Python code block — no explanation, no prose.

```python
<solution here>
```
"""

VOTING_INPUT_PROMPT_DRHALL = """\
You are an expert Python programmer designing test INPUTS for a LeetCode problem.

Given the problem description and the function signature in the starter code,
generate {n} diverse, VALID input argument lists for calling the solution method.

Rules:
- Cover normal cases, boundary cases (smallest valid size), and edge cases,
  all within the problem's stated constraints.
- Each line must contain ONLY the argument list, written exactly as it would
  appear inside the parentheses of a Python call, using keyword arguments.
- Do NOT include expected outputs.  Do NOT include the function name.
- Do NOT add explanations, numbering, or code fences.

Example output format (for a method def twoSum(self, nums, target)):
nums = [2, 7, 11, 15], target = 9
nums = [3, 3], target = 6
nums = [1, 2], target = 3

Now output exactly {n} lines:
"""

MUTATION_PROMPT_NOMR = """Given a question and a base response, rewrite the base response as a single complete 
sentence that answers the question. Express it in your own words. If the base response is incomplete, rewrite it 
into a full sentence using the question's context. Output only the rewritten sentence, with no explanation, no 
numbering, and no extra text."""
