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

3. Polarity Transformation
   Introduce or remove a pair of logical negations while preserving the \
original assertion. For example, "Paris is the capital of France." becomes \
"It is not true that Paris is not the capital of France." Keep all entities, \
dates, quantities, and conditions unchanged. Do not reverse the claim using \
a single negation, an antonym, or a changed qualification.

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

def qa_mutation_prompt(n: int = 5) -> str:
    """Return the shared MR-guided QA mutation prompt for every model."""
    return MUTATION_PROMPT.format(n=n)

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

REFINE_PROMPT = """You are a strict answer extractor. The input is the raw output of
an answer-selection step and may contain a pairwise-ranking matrix, candidate
numbers, confidence scores, intermediate reasoning, explanations, or formatting
noise in addition to the selected answer.

Extract the answer that the input identifies as the final or best answer. Do not
verify, correct, re-rank, reinterpret, or improve it. If no final answer is stated
explicitly, use only the selection information already present in the input: choose
the candidate with the best overall rank (lowest total rank) or, for confidence
selection, the highest confidence score. When an ORIGINAL CANDIDATES section is
provided, use it only to copy the already-selected numbered candidate verbatim.

Return exactly one complete answer sentence and nothing else. Copy the selected
answer verbatim whenever possible, removing only surrounding labels, candidate
numbers, quotation marks, or formatting. Never return a ranking matrix, scores,
analysis, explanation, prefix, or commentary."""

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

3. Polarity Transformation
   Introduce or remove double negation in a Boolean condition while preserving \
its truth value, e.g. `if condition` → `if not not condition` or \
`if x < y` → `if not not (x < y)`. Keep branch bodies, comparisons, \
boundary values, and return values unchanged. Apply this only where Boolean \
semantics are preserved; do not replace a value-producing expression with \
a Boolean or change the program's behavior.

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
You are an impartial judge ranking {n} Python solutions for one LeetCode problem.
In this single evaluation, compare every ordered candidate pair and construct an
{n}-by-{n} score matrix R.

Scoring criteria (in priority order):
1. Functional correctness, including all stated edge cases.
2. Absence of hallucinated functions, methods, modules, or language features.
3. Algorithmic soundness and efficiency.
4. Code quality and completeness.

For every i != j, R[i][j] must be an integer in [1, {n}] that scores Candidate
i relative to Candidate j: lower is better. Set every diagonal entry R[i][i] to
0. Apply the same scale consistently across all pairs. The winning candidate is
the row with the lowest sum; ties are resolved by the lowest candidate number.

Return JSON only, with no Markdown or explanation:
{{"score_matrix": [[0, 1], [2, 0]]}}
The actual matrix must contain exactly {n} rows and {n} integers per row.
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
