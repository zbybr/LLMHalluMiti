import time
import re
from difflib import SequenceMatcher
from copy import deepcopy
import spacy
import Levenshtein
from utils import answer_by_model_key_with_cost

SHOW = True
sleep_time = 0
threshold = 0.8
nlp = spacy.load("en_core_web_trf")


def get_entity(reasoning_path):
    """Identification of significant entity and its category"""
    pattern = r'The most relevant entity is:? (.*?)and its'
    result = re.findall(pattern, reasoning_path, re.IGNORECASE)
    if len(result) != 0:
        entity = result[-1]
    else:
        entity = 'None'  # No match found.
    entity = entity.replace("\"", "").replace("*", "").strip()
    return entity


def get_category(reasoning_path):
    """Identification of significant entity and its category"""
    pattern = r'and its category is:? (.*?)\.'
    result = re.findall(pattern, reasoning_path, re.IGNORECASE)
    if len(result) != 0:
        category = result[-1]
    else:
        category = 'None'  # No match found.
    category = category.replace("\"", "").replace("*", "").strip()
    return category


def get_answer(reasoning_path):
    """Get Answer from the Reasoning Path"""
    pattern = r'The answer is:? (.*?)\.'
    result = re.findall(pattern, reasoning_path, re.IGNORECASE)
    if len(result) != 0:
        answer = result[-1]
    else:
        answer = 'None'  # No match found.
    answer = answer.replace("\"", "").replace("*", "").strip()
    return answer


def get_verification_question_answer(reasoning_path):
    """Get Answer from the Reasoning Path"""
    # pattern = r'[?\"?X\"?]? refers to:? (.*?)\.'
    pattern = r'[?\"?X\"?]? refers to:? (.*?)$'
    result = re.findall(pattern, reasoning_path, re.IGNORECASE)
    if len(result) != 0:
        answer = result[-1]
    else:
        answer = 'None'  # No match found.
    answer = answer.replace("\"", "").replace("*", "").strip()
    if answer[-1] == '.':
        answer = answer[:-1]
    return answer


def identify_important_entity(process_record, model_key, question):
    state = True
    prompt = f"""Define: Named-entity recognition seeks to locate and classify named entities mentioned in a question into pre-defined categories such as person names, 
        organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc. If an entity does not fit the types above it is (other). 
        Adjectives and verbs are not entities. 

        Instruction: Given the question below, identify a list of possible entities within the question and for each entry explain why it either is or is not an entity, and select the most 
        relevant entity to the question from that list. The reasoning process ends with the conclusion \"The most relevant entity is (entity) and its category is (category),\" 
        e.g., \"The most relevant entity is Notre Dame and its category is location.\"

        Question: {question} 

        Answer: 
        """
    response, tokens = answer_by_model_key_with_cost(
        prompt=prompt,
        model_key=model_key
    )
    time.sleep(sleep_time)
    if SHOW:
        print(f'\n[INFO]:\t\tEntity Identification Process: {response}')
    entity = get_entity(response)
    category = get_category(response)
    if SHOW:
        print(f'\n[INFO]:\t\tEntity: {entity}, Category: {category}')
    similarity_ratio = SequenceMatcher(None, entity.lower(), question.lower()).ratio()
    if entity.lower() in question.lower():
        pass
    elif entity == 'None' or category == 'None' or similarity_ratio < threshold:
        try:
            doc = nlp(question)
            entity_category = [(ent.text, ent.label_) for ent in doc.ents]
            entity, category = entity_category[-1]
        except Exception as e:
            state = False
    elif entity.lower() not in question.lower():
        s = SequenceMatcher(None, entity.lower(), question.lower())
        match = s.find_longest_match(0, len(entity), 0, len(question))
        entity = entity[match.a: match.a + match.size]
    if state:
        process_record['entity_category'] = {}
        process_record['entity_category']['reasoning_path'] = response
        process_record['entity_category']['entity'] = entity
        process_record['entity_category']['category'] = category
        process_record['entity_category']['token_cost'] = tokens
    return entity.lower(), category.lower(), process_record, state, tokens


def construct_verification_question_pro(question, entity):
    """
        {replace the selected {entity} in {question} with X}
    """
    verification_question = question.lower().replace(entity, "[X]")
    return verification_question


def generate_document(process_record, model_key, question, num_iter, flag):
    """Generate Document"""
    prompt = f"""Generate a background document to answer the given question.\n\n{question}\n\n"""
    document, tokens = answer_by_model_key_with_cost(
        prompt=prompt,
        model_key=model_key,
    )
    if SHOW:
        print(f'\n[INFO]:\t\tDocument: {document}')
    time.sleep(sleep_time)
    if flag == 'init':
        process_record[f'{num_iter}-iter'][f'{flag}-document'] = document
        process_record[f'{num_iter}-iter'][f'{flag}-tokens'] = tokens
    else:
        process_record[f'{num_iter}-iter']['rectification'] = {}
        process_record[f'{num_iter}-iter']['rectification'][f'{flag}-document'] = document
        process_record[f'{num_iter}-iter']['rectification'][f'{flag}-tokens'] = tokens
    return document, tokens


def generate_answer(entity, process_record, model_key, question, num_iter, document, flag):
    if flag == 'init':
        answer = entity
        tokens = 0
    else:
        prompt = f"""Refer to the passage below and answer the following question with just one entity.\n\nPassage: {document}\n\nQuestion: {question}\n\nThe answer is"""
        answer, tokens = answer_by_model_key_with_cost(
            prompt=prompt,
            model_key=model_key,
        )
    if SHOW:
        print(f'\n[INFO]:\t\tanswer: {answer}')
    answer = str(answer)
    time.sleep(sleep_time)
    if flag == 'init':
        process_record[f'{num_iter}-iter'][f'{flag}-answer'] = answer
        process_record[f'{num_iter}-iter'][f'{flag}-tokens'] = tokens
    else:
        process_record[f'{num_iter}-iter']['rectification'][f'{flag}-answer'] = answer
        process_record[f'{num_iter}-iter']['rectification'][f'{flag}-tokens'] = tokens
    return answer, tokens


def construct_verification_question(verification_question_pro, answer):
    """
        {replace the selected {entity} in {question} with X} Suppose the answer is {answer}. What is X?
    """
    verification_question = verification_question_pro + f' Suppose the answer is {answer}. What is [X]?'
    return verification_question


def solve_verification_question(process_record, model_key, num_iter, verification_question, category):
    prompt = f"""Solve an open domain question. The reasoning path ends with \"The answer is (entity)\", e.g. \"The answer is Notre Dame\".

        Question: {verification_question}

        Answer: The category of X is {category}. Let's think step by step.
        """
    reasoning_path, tokens = answer_by_model_key_with_cost(
        prompt=prompt,
        model_key=model_key
    )
    if SHOW:
        print(f'\n[INFO]:\t\tVerification Question Reasoning Path: {reasoning_path}')
    time.sleep(sleep_time)
    answer = get_verification_question_answer(reasoning_path)
    if SHOW:
        print(f'\n[INFO]:\t\tVerification Question Answer: {answer}')
    process_record[f'{num_iter}-iter']['verification'] = {}
    process_record[f'{num_iter}-iter']['verification']['verification_question'] = verification_question
    process_record[f'{num_iter}-iter']['verification']['verification_question_reasoning_path'] = reasoning_path
    process_record[f'{num_iter}-iter']['verification']['verification_question_answer'] = answer
    process_record[f'{num_iter}-iter']['verification']['verification_question_token_cost'] = tokens
    return answer, tokens


def verification_result(process_record, model_key, num_iter, verification_question, entity, entity_prediction):
    prompt = f"""
        Instruction: Determine whether the proposition is correct or incorrect. The reasoning path ends with \"The result of the judgment is (the result of the judgment)\", e.g. \"The result of the judgment is correct\". 
        Proposition: If the answer to the question \"{verification_question}\" is \"{entity}\", then X could also be \"{entity_prediction}\"

        A: Let's think step by step.
        """
    judgement_process, tokens = answer_by_model_key_with_cost(
        prompt=prompt,
        model_key=model_key,
    )
    if SHOW:
        print(f'\n[INFO]:\t\tJudgement Process: {judgement_process}')
    time.sleep(sleep_time)
    process_record[f'{num_iter}-iter']['judgement'] = {}
    process_record[f'{num_iter}-iter']['judgement']['judgement_process'] = judgement_process
    process_record[f'{num_iter}-iter']['judgement']['token_cost'] = tokens
    if "incorrect" in judgement_process.lower():
        process_record[f'{num_iter}-iter']['judgement']['judgement_result'] = "incorrect"
        return False, tokens
    else:
        process_record[f'{num_iter}-iter']['judgement']['judgement_result'] = "correct"
        return True, tokens


def rectified_question(question, incorrect_answer_record):
    """
        {question} (The answer is likely not in {potentially incorrect answers})
    """
    refined_question = f"{question} (The answer is likely not in list {incorrect_answer_record})"
    return refined_question


def pipeline(question, base_response, process_record, model_key, max_iteration):
    """pipline with token used"""
    total_tokens_used = 0
    start_pipeline = time.time()

    if SHOW:
        print(f'\n[INFO]:\t\tQuestion: {question}')

    answer_record, incorrect_answer_record = [], []
    process_record[f'0-iter'] = {}
    entity, category, process_record, state, tokens = identify_important_entity(process_record, model_key, question)
    total_tokens_used += tokens
    verification_question_pro = construct_verification_question_pro(question, entity)
    document, tokens = generate_document(process_record, model_key, question, 0, 'init')
    total_tokens_used += tokens
    answer, tokens = generate_answer(base_response, process_record, model_key, question, 0, document, 'init')
    total_tokens_used += tokens
    # base_response = answer
    answer_record.append(answer)
    if state:
        for num_iter in range(max_iteration):
            num_iter += 1
            process_record[f'{num_iter}-iter'] = {}
            verification_question = construct_verification_question(verification_question_pro, answer_record[-1])
            entity_prediction, tokens = solve_verification_question(process_record, model_key, num_iter,
                                                                    verification_question, category)
            total_tokens_used += tokens
            judgement, tokens = verification_result(process_record, model_key, num_iter, verification_question, entity,
                                                    entity_prediction)
            total_tokens_used += tokens
            if judgement:
                break
            elif Levenshtein.distance(entity_prediction.lower(),
                                      entity.lower()) <= 5 or entity_prediction.lower() in entity.lower() or entity.lower() in entity_prediction.lower():
                break
            else:
                incorrect_answer_record = deepcopy(answer_record)
                refined_question = rectified_question(question, incorrect_answer_record)
                refined_document, tokens = generate_document(process_record, model_key, refined_question, num_iter,
                                                             'refined')
                total_tokens_used += tokens
                refined_answer, tokens = generate_answer("", process_record, model_key, refined_question, num_iter,
                                                         refined_document, 'refined')
                total_tokens_used += tokens
                answer_record.append(refined_answer)
            if len(answer_record) >= 2:
                answer_record_history = [history.lower() for history in answer_record[:-1]]
                if answer_record[-1].lower() in answer_record_history:
                    break

    final_answer = answer_record[-1]
    total_pipeline_time = time.time() - start_pipeline
    process_record['total_tokens_used'] = total_tokens_used
    process_record['total_pipeline_time'] = total_pipeline_time

    return final_answer, process_record, total_tokens_used, total_pipeline_time
