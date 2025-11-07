from langchain_classic.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
import prompts
from execute_verification_chain import ExecuteVerificationChain


class WikiDataCategoryListCOVEChain(object):
    def __init__(self, llm, external_baseline_response=None):
        self.llm = llm
        self.external_baseline_response = external_baseline_response

    def __call__(self):
        chains_list = []
        # Create baseline response if no external_baseline_response
        if not self.external_baseline_response:
            baseline_response_prompt_template = PromptTemplate(
                input_variables=["original_question"],
                template=prompts.BASELINE_PROMPT_WIKI
            )
            baseline_response_chain = LLMChain(
                llm=self.llm,
                prompt=baseline_response_prompt_template,
                output_key="baseline_response"
            )
            chains_list.append(baseline_response_chain)
        # Create plan verification
        verification_question_template_prompt_template = PromptTemplate(input_variables=["original_question"],
                                                                        template=prompts.VERIFICATION_QUESTION_TEMPLATE_PROMPT_WIKI)
        verification_question_template_chain = LLMChain(llm=self.llm,
                                                        prompt=verification_question_template_prompt_template,
                                                        output_key="verification_question_template")
        # Create plan verification questions
        verification_question_generation_prompt_template = PromptTemplate(input_variables=["original_question",
                                                                                           "baseline_response",
                                                                                           "verification_question_template"],
                                                                          template=prompts.VERIFICATION_QUESTION_PROMPT_WIKI)
        verification_question_generation_chain = LLMChain(llm=self.llm,
                                                          prompt=verification_question_generation_prompt_template,
                                                          output_key="verification_questions")
        # Create execution verification
        execute_verification_question_prompt_template = PromptTemplate(input_variables=["verification_questions"],
                                                                       template=prompts.EXECUTE_PLAN_PROMPT)
        execute_verification_question_chain = ExecuteVerificationChain(llm=self.llm,
                                                                       prompt=execute_verification_question_prompt_template,
                                                                       output_key="verification_answers")
        # Create final refined response
        final_answer_prompt_template = PromptTemplate(input_variables=["original_question",
                                                                       "baseline_response",
                                                                       "verification_answers"],
                                                      template=prompts.FINAL_REFINED_PROMPT)
        final_answer_chain = LLMChain(llm=self.llm,
                                      prompt=final_answer_prompt_template,
                                      output_key="final_answer")

        chains_list.append(verification_question_template_chain)
        chains_list.append(verification_question_generation_chain)
        chains_list.append(execute_verification_question_chain)
        chains_list.append(final_answer_chain)

        input_vars = ["original_question"] if not self.external_baseline_response else ["original_question",
                                                                                        "baseline_response"]
        # Create sequential chain
        wiki_data_category_list_cove_chain = SequentialChain(chains=chains_list,
                                                             input_variables=input_vars,
                                                             # Here we return multiple variables
                                                             output_variables=["original_question",
                                                                               "baseline_response",
                                                                               "verification_question_template",
                                                                               "verification_questions",
                                                                               "verification_answers",
                                                                               "final_answer"],
                                                             verbose=False)
        return wiki_data_category_list_cove_chain


class MultiSpanCOVEChain(object):
    def __init__(self, llm, external_baseline_response=None):
        self.llm = llm
        self.external_baseline_response = external_baseline_response

    def __call__(self):
        chains_list = []
        # Create baseline response chain if no external_baseline_response
        if not self.external_baseline_response:
            baseline_response_prompt_template = PromptTemplate(
                input_variables=["original_question"],
                template=prompts.BASELINE_PROMPT_WIKI
            )
            baseline_response_chain = LLMChain(
                llm=self.llm,
                prompt=baseline_response_prompt_template,
                output_key="baseline_response"
            )
            chains_list.append(baseline_response_chain)
        # Create plan verification questions
        verification_question_generation_prompt_template = PromptTemplate(input_variables=["original_question",
                                                                                           "baseline_response"],
                                                                          template=prompts.VERIFICATION_QUESTION_PROMPT_MULTI)
        verification_question_generation_chain = LLMChain(llm=self.llm,
                                                          prompt=verification_question_generation_prompt_template,
                                                          output_key="verification_questions")
        # Create execution verification
        execute_verification_question_prompt_template = PromptTemplate(input_variables=["verification_questions"],
                                                                       template=prompts.EXECUTE_PLAN_PROMPT)
        execute_verification_question_chain = ExecuteVerificationChain(llm=self.llm,
                                                                       prompt=execute_verification_question_prompt_template,
                                                                       output_key="verification_answers")
        # Create final refined response
        final_answer_prompt_template = PromptTemplate(input_variables=["original_question",
                                                                       "baseline_response",
                                                                       "verification_answers"],
                                                      template=prompts.FINAL_REFINED_PROMPT)
        final_answer_chain = LLMChain(llm=self.llm,
                                      prompt=final_answer_prompt_template,
                                      verbose=True,
                                      output_key="final_answer")

        chains_list.append(verification_question_generation_chain)
        chains_list.append(execute_verification_question_chain)
        chains_list.append(final_answer_chain)
        input_vars = ["original_question"] if not self.external_baseline_response else ["original_question",
                                                                                        "baseline_response"]
        # Create sequential chain
        multi_span_cove_chain = SequentialChain(chains=chains_list,
                                                input_variables=input_vars,
                                                # Here we return multiple variables
                                                output_variables=["original_question",
                                                                  "baseline_response",
                                                                  "verification_questions",
                                                                  "verification_answers",
                                                                  "final_answer"],
                                                verbose=False)
        return multi_span_cove_chain


class LongFormCOVEChain(object):
    def __init__(self, llm, external_baseline_response=None):
        self.llm = llm
        self.external_baseline_response = external_baseline_response

    def __call__(self):
        chains_list = []
        # Create baseline response chain
        if not self.external_baseline_response:
            baseline_response_prompt_template = PromptTemplate(
                input_variables=["original_question"],
                template=prompts.BASELINE_PROMPT_WIKI
            )
            baseline_response_chain = LLMChain(
                llm=self.llm,
                prompt=baseline_response_prompt_template,
                output_key="baseline_response"
            )
            chains_list.append(baseline_response_chain)
        verification_question_generation_prompt_template = PromptTemplate(input_variables=["original_question",
                                                                                           "baseline_response"],
                                                                          template=prompts.VERIFICATION_QUESTION_PROMPT_LONG)
        verification_question_generation_chain = LLMChain(llm=self.llm,
                                                          prompt=verification_question_generation_prompt_template,
                                                          output_key="verification_questions")
        # Create execution verification
        execute_verification_question_prompt_template = PromptTemplate(input_variables=["verification_questions"],
                                                                       template=prompts.EXECUTE_PLAN_PROMPT)
        execute_verification_question_chain = ExecuteVerificationChain(llm=self.llm,
                                                                       prompt=execute_verification_question_prompt_template,
                                                                       output_key="verification_answers")
        # Create final refined response
        final_answer_prompt_template = PromptTemplate(input_variables=["original_question",
                                                                       "baseline_response",
                                                                       "verification_answers"],
                                                      template=prompts.FINAL_REFINED_PROMPT)
        final_answer_chain = LLMChain(llm=self.llm,
                                      prompt=final_answer_prompt_template,
                                      verbose=True,
                                      output_key="final_answer")
        chains_list.append(verification_question_generation_chain)
        chains_list.append(execute_verification_question_chain)
        chains_list.append(final_answer_chain)
        input_vars = ["original_question"] if not self.external_baseline_response else ["original_question",
                                                                                        "baseline_response"]
        # Create sequential chain
        long_form_cove_chain = SequentialChain(chains=chains_list,
                                               input_variables=input_vars,
                                               # Here we return multiple variables
                                               output_variables=["original_question",
                                                                 "baseline_response",
                                                                 "verification_questions",
                                                                 "verification_answers",
                                                                 "final_answer"],
                                               verbose=False)
        return long_form_cove_chain
