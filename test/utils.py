import tiktoken
import spacy
from sentence_transformers import SentenceTransformer, util
encoding_name = "cl100k_base"
encoding = tiktoken.get_encoding(encoding_name)
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def num_tokens_from_string(s):
    return len(encoding.encode(s))


def check_string(s):
    if s.strip() == "" or s.strip() == "IP access frequency is too high, please try again later":
        raise ValueError("Empty or invalid string encountered.")


def extract_facts(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def compute_factscore(final_answer, correct_answer, threshold=0.8):
    final_facts = extract_facts(final_answer)
    correct_facts = extract_facts(correct_answer)

    if not final_facts:
        return 0.0

    final_embeddings = model.encode(final_facts, convert_to_tensor=True)
    correct_embeddings = model.encode(correct_facts, convert_to_tensor=True)

    correct_count = 0
    for i, fact in enumerate(final_facts):
        sims = util.cos_sim(final_embeddings[i], correct_embeddings)[0]
        if sims.max().item() >= threshold:
            correct_count += 1

    return correct_count / len(final_facts)


# question = "Who wrote Hamlet?"
# correct_answer = "Hamlet was written by William Shakespeare."
# final_answer = "Hamlet was authored by William Shakespeare in London."
#
# score = compute_factscore(final_answer, correct_answer)
# print(f"FactScore: {score:.2f}")