import re
from tqdm import tqdm
from typing import List, Tuple, Dict
import math
import numpy as np
import copy
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from flashrag.utils import get_retriever, get_generator, selfask_pred_parse, ircot_pred_parse
from flashrag.pipeline import BasicPipeline
from flashrag.dataset.utils import get_batch_dataset, merge_batch_dataset
from flashrag.prompt import PromptTemplate
from flashrag.utils.utils import extract_between

class ReasoningPipeline(BasicPipeline):
    system_prompt=""
    user_prompt=(
        "The User asks a question, and the Assistant solves it.\n"
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.\n"
        "The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, "
        "respectively, i.e., \"<think> reasoning process here </think>\\n\\n<answer> final answer here </answer>\".\n"
        "During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "
        "\"<|begin_of_query|> search query (only keywords) here <|end_of_query|>\". **A query must involve only a single triple**.\n"
        "Then, the system will provide the Assistant with helpful information with the format of "
        "\"<|begin_of_documents|> ...search results... <|end_of_documents|>\".\n\n"
        "User:{question}\n"
        "Assistant: <think>"
    )


    def __init__(self, config, 
        prompt_template=None, 
        max_retrieval_num=5, 
        begin_of_query_token="<|begin_of_query|>",
        end_of_query_token="<|end_of_query|>",
        begin_of_documents_token="<|begin_of_documents|>",
        end_of_documents_token="<|end_of_documents|>",
        begin_of_answer_token="<answer>",
        end_of_answer_token='</answer>',
        retriever=None, 
        generator=None
    ):
        if prompt_template is None:
            prompt_template = PromptTemplate(
                config=config,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt
            )
        super().__init__(config, prompt_template)

        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator
        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        self.max_retrieval_num = max_retrieval_num

        self.begin_of_query_token = begin_of_query_token
        self.end_of_query_token = end_of_query_token
        self.begin_of_documents_token = begin_of_documents_token
        self.end_of_documents_token = end_of_documents_token
        self.begin_of_answer_token = begin_of_answer_token
        self.end_of_answer_token = end_of_answer_token

        self.stop_tokens = ["<|im_end|>", "<|endoftext|>", self.end_of_answer_token, self.end_of_query_token]
    
    def _retrieved_docs_to_string(self, retrieved_docs: List[Dict]):
        format_doc_string = ""
        for idx, doc in enumerate(retrieved_docs):
            contents = doc['contents']
            title = contents.split('\n')[0]
            text = '\n'.join(contents.split('\n')[1:])
            doc_string = f"Title: {title} Text: {text}"
            doc_string = re.sub(r'^\d+\s+', '', doc_string)
            format_doc_string += f'({idx+1}){doc_string}\n'
        format_doc_string = f'\n\n{self.begin_of_documents_token}\n{format_doc_string}\n{self.end_of_documents_token}\n\n'
        return format_doc_string

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        prompts = [self.prompt_template.get_string(question=question) for question in dataset.question]
        dataset.update_output('prompt', prompts)
        dataset.update_output('finish_flag', [False] * len(prompts))
        dataset.update_output('retrieval_results', [{} for _ in range(len(prompts))])
        dataset.update_output('retrieved_times', [0] * len(prompts))

        # Logic of reasoning
        for current_step_idx in range(self.max_retrieval_num + 1):
            exist_items = [item for item in dataset if item.finish_flag == False]
            exist_prompts = [item.prompt for item in exist_items]
            
            print(f"Current step: {current_step_idx}, exist_items: {len(exist_items)}")

            if len(exist_items) == 0:
                print("All prompts are finished")
                break
            if current_step_idx == self.max_retrieval_num:
                print("Max retrieval number reached")
                for item in exist_items:
                    item.pred = 'No valid answer found'
                    item.finish_flag = True
                    item.finish_reason = 'Reach max retrieval number'
                break

            step_outputs = self.generator.generate(exist_prompts, stop=self.stop_tokens)
            step_query_list = [] # store generated queries for retrieval

            # parse each sample's step output
            for item, step_output in zip(exist_items, step_outputs):
                item.prompt = item.prompt + step_output.strip()
                if self.end_of_answer_token in step_output and step_output.endswith(self.end_of_answer_token):
                    item.pred = str(extract_between(step_output, self.begin_of_answer_token, self.end_of_answer_token))
                    item.finish_flag = True
                    item.finish_reason = "Finished"
                
                elif self.begin_of_query_token in step_output and step_output.endswith(self.end_of_query_token):
                    query = extract_between(step_output, self.begin_of_query_token, self.end_of_query_token)
                    if query is not None:
                        step_query_list.append({'item': item, 'query': query})
                    else:
                        item.pred = 'No valid answer found'
                        item.finish_flag = True
                        item.finish_reason = 'Query instruction error'

                else:
                    item.pred = step_output.strip()
                    item.finish_flag = True
                    item.finish_reason = 'Normal finish without answer pattern'
                
            # do retrieval and add retrieved docs to prompt
            if len(step_query_list) > 0:
                retrieved_docs = self.retriever.batch_search([it['query'] for it in step_query_list])
                for it, item_retrieved_docs in zip(step_query_list, retrieved_docs):
                    item = it['item']
                    query = it['query']
                    item.retrieval_results[item.retrieved_times] = {'query': query,'docs': copy.copy(item_retrieved_docs)}
                    #item.retrieved_docs += [item_retrieved_docs]
                    format_doc_string = self._retrieved_docs_to_string(item_retrieved_docs)
                    item.prompt += format_doc_string
                    item.retrieved_times += 1

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

            
                
                        

                
            
        
            