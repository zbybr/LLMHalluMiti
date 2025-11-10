import tiktoken
import spacy
from sentence_transformers import SentenceTransformer, util
encoding_name = "cl100k_base"
encoding = tiktoken.get_encoding(encoding_name)


def num_tokens_from_string(s):
    return len(encoding.encode(s))


def check_string(s):
    if s.strip() == "" or s.strip() == "IP access frequency is too high, please try again later":
        raise ValueError("Empty or invalid string encountered.")