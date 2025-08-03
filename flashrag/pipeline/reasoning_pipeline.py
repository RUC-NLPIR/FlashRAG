import re
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Union
import math
import json
import numpy as np
import copy
# from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from flashrag.utils import get_retriever, get_generator, selfask_pred_parse, ircot_pred_parse
from flashrag.pipeline import BasicPipeline
from flashrag.pipeline.ReaRAG_utils import AgentUtils
from flashrag.dataset.utils import get_batch_dataset, merge_batch_dataset
from flashrag.prompt import PromptTemplate
from flashrag.prompt import get_generate_final_answer_message,get_generate_intermediate_answer_message,get_generate_subquery_message
from flashrag.utils.utils import extract_between,extract_between_all


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
        generator=None,
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
                if self.end_of_answer_token in step_output and (step_output.endswith(self.end_of_answer_token) or step_output.endswith("<|endoftext|>")):
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


class SimpleDeepSearcherPipeline(ReasoningPipeline):
    system_prompt=""
    user_prompt=(
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        "Whenever you encounter a topic, fact, or piece of information you are uncertain about or need further details on, please perform a search to gather more accurate, up-to-date, or specific information. You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n"
        "- Do not generate <|begin_search_result|> and <|end_search_result|> tags yourself.\n\n"
        "Please answer the following question. You should think step by step to solve it.\n\n"
        "Provide your final answer in the format \\boxed{{YOUR_ANSWER}}.\n\n"
        "Question:\n{question}\n\n"
    )
    def __init__(self, config,
        prompt_template=None, 
        max_retrieval_num=10, 
        begin_of_query_token="<|begin_search_query|>",
        end_of_query_token="<|end_search_query|>",
        begin_of_documents_token="<|begin_search_result|>",
        end_of_documents_token="<|end_search_result|>",
        begin_of_answer_token=None,
        end_of_answer_token=None,
        retriever=None, 
        generator=None,
    ):
        super().__init__(config,
                         prompt_template=prompt_template,
                         max_retrieval_num=max_retrieval_num, 
                         begin_of_query_token=begin_of_query_token,
                         end_of_query_token=end_of_query_token,
                         begin_of_documents_token=begin_of_documents_token,
                         end_of_documents_token=end_of_documents_token,
                         begin_of_answer_token=begin_of_answer_token,
                         end_of_answer_token=end_of_answer_token,
                         retriever=retriever, 
                         generator=generator,
                         )
        self.stop_tokens = ["<|im_end|>", "<|endoftext|>", self.end_of_query_token]

    def truncate_prompt(self, prompt ,max_len):
            assert isinstance(prompt, str)
            try:
                tokenized_prompt = self.prompt_template._get_tokenizer().encode(prompt, truncation=False, return_tensors="pt").input_ids[0]
            except:
                tokenized_prompt = self.prompt_template._get_tokenizer().encode(prompt, truncation=False, return_tensors="pt")[0]

            if len(tokenized_prompt) >= max_len:
                print(f"The doc length is greater than the maximum length ({len(tokenized_prompt)} > {max_len}) and has been truncated!")
                half = int(max_len / 2) - 20
                prompt = self.prompt_template._get_tokenizer().decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                        self.prompt_template._get_tokenizer().decode(tokenized_prompt[-half:], skip_special_tokens=True)
            return prompt

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        prompts = [self.prompt_template.get_string(MAX_SEARCH_LIMIT=self.max_retrieval_num, question=question) for question in dataset.question]
        dataset.update_output('prompt', prompts)
        dataset.update_output('finish_flag', [False] * len(prompts))
        dataset.update_output('retrieval_results', [{} for _ in range(len(prompts))])
        dataset.update_output('retrieved_times', [0] * len(prompts))
        
        for current_step_idx in range(self.max_retrieval_num + 1):
            exist_items = [item for item in dataset if item.finish_flag == False]
            exist_prompts = [self.prompt_template.truncate_prompt(item.prompt) for item in exist_items]
            
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
                if "\\boxed" in step_output:
                    item.pred = str(extract_between(step_output, "\\boxed{", "}"))
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

            if len(step_query_list) > 0:
                retrieved_docs = self.retriever.batch_search([it['query'] for it in step_query_list])
                for it, item_retrieved_docs in zip(step_query_list, retrieved_docs):
                    item = it['item']
                    query = it['query']
                    item.retrieval_results[item.retrieved_times] = {'query': query,'docs': copy.copy(item_retrieved_docs)}
                    format_doc_string = self.truncate_prompt(self._retrieved_docs_to_string(item_retrieved_docs),3000)
                    item.prompt += format_doc_string
                    item.retrieved_times += 1

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset  

class SearchR1Pipeline(ReasoningPipeline):
    system_prompt=""
    user_prompt=(
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
    )
    
    def __init__(self, config,
        prompt_template=None, 
        max_retrieval_num=5, 
        begin_of_query_token="<search>",
        end_of_query_token="</search>",
        begin_of_documents_token="<information>",
        end_of_documents_token="</information>",
        begin_of_answer_token="<answer>",
        end_of_answer_token='</answer>',
        retriever=None, 
        generator=None,
    ):
        super().__init__(config,
                         prompt_template=prompt_template,
                         max_retrieval_num=max_retrieval_num, 
                         begin_of_query_token=begin_of_query_token,
                         end_of_query_token=end_of_query_token,
                         begin_of_documents_token=begin_of_documents_token,
                         end_of_documents_token=end_of_documents_token,
                         begin_of_answer_token=begin_of_answer_token,
                         end_of_answer_token=end_of_answer_token,
                         retriever=retriever, 
                         generator=generator,
                         )
        self.stop_tokens = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n", "</answer>", "</answer>\n", "</answer>\n\n","<|endoftext|>","<|im_end|>"]

    def _retrieved_docs_to_string(self, retrieved_docs: List[Dict]):
        format_doc_string = ""
        for idx, doc in enumerate(retrieved_docs):
            contents = doc['contents']
            title = contents.split('\n')[0]
            text = '\n'.join(contents.split('\n')[1:])
            format_doc_string += f"Doc {idx+1}(Title: {title}) {text}\n"
        format_doc_string = f'\n\n{self.begin_of_documents_token}\n{format_doc_string}\n{self.end_of_documents_token}\n\n'
        return format_doc_string       
    
    
class AutoRefinePipeline(SearchR1Pipeline):
    system_prompt=""
    user_prompt = (
        "You are a helpful assistant who is good at answering questions with multi-turn search engine calling. "
        "To answer questions, you must first reason through the available information using <think> and </think>. "
        "If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. "
        "The retrieval system will provide you with the three most relevant documents enclosed in<documents> and </documents>. "
        "After each search, you need to summarize and refine the existing documents in <refine> and </refine>. You may send multiple search requests if needed. "
        "Once you have sufficient information, provide a concise final answer using <answer> and </answer>.\n"
        "<user> Question: {question} </user>"
    )
    
    
    def __init__(self, config,
        prompt_template=None, 
        max_retrieval_num=5, 
        begin_of_query_token="<search>",
        end_of_query_token="</search>",
        begin_of_documents_token="<documents>",
        end_of_documents_token="</documents>",
        begin_of_answer_token="<answer>",
        end_of_answer_token='</answer>',
        retriever=None, 
        generator=None,
    ):
        super().__init__(config,
                         prompt_template=prompt_template,
                         max_retrieval_num=max_retrieval_num, 
                         begin_of_query_token=begin_of_query_token,
                         end_of_query_token=end_of_query_token,
                         begin_of_documents_token=begin_of_documents_token,
                         end_of_documents_token=end_of_documents_token,
                         begin_of_answer_token=begin_of_answer_token,
                         end_of_answer_token=end_of_answer_token,
                         retriever=retriever, 
                         generator=generator,
                         )
        self.stop_tokens = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n", "</answer>", "</answer>\n", "</answer>\n\n","<|endoftext|>","<|im_end|>"]    
            
            
class O2SearcherPipeline(ReasoningPipeline):
    system_prompt=(
        "As a expert researcher, provide comprehensive key findings for open-ended queries and precise answers to other specific questions. "
        "Each time you receive new information, you MUST first engage in reasoning within the <think> and </think> tags. "
        "After reasoning, if you realize that you lack certain knowledge, you can invoke a SEARCH action with distinct queries (one to five) using the <search>\n<query>QUERY</query>\n<query>QUERY</query>\n</search> format to obtain relevant learnings, "
        "which will be presented between the <learnings> and </learnings> tags.\n "
        "You are allowed to perform searches as many times as necessary. If you determine that no additional external knowledge is required, you can directly present the output within the <answer> and </answer> tags."
    )
    user_prompt='''Initial query:{question}'''
    error_prompt='''The response you attempted before is invalid. If you plan to execute actions like SEARCH, you need to enclose the SEARCH queries within the <search> and </search> tags. Furthermore, the required queries for the SEARCH action should be placed between the <query> and </query> tags. Moreover, if you wish to present the final output for the initial query, you must wrap the result within the <answer> and </answer> tags.
'''
    extra_prompt='''Search learnings: <learnings>{learning_str}</learnings>.
'''
    
    def __init__(self, config,
        prompt_template=None, 
        max_retrieval_num=5, 
        begin_of_query_token="<query>",
        end_of_query_token="</query>",
        begin_of_search_token="<search>",
        end_of_search_token="</search>",
        begin_of_documents_token="<learnings>",
        end_of_documents_token="</learnings>",
        begin_of_answer_token="<answer>",
        end_of_answer_token='</answer>',
        retriever=None, 
        generator=None,
    ):
        super().__init__(config,
                         prompt_template=prompt_template,
                         max_retrieval_num=max_retrieval_num, 
                         begin_of_query_token=begin_of_query_token,
                         end_of_query_token=end_of_query_token,
                         begin_of_documents_token=begin_of_documents_token,
                         end_of_documents_token=end_of_documents_token,
                         begin_of_answer_token=begin_of_answer_token,
                         end_of_answer_token=end_of_answer_token,
                         retriever=retriever, 
                         generator=generator,
                         )
        self.begin_of_search_token = begin_of_search_token
        self.end_of_search_token = end_of_search_token
        self.tokenizer = self.prompt_template._get_tokenizer()
        self.stop_tokens = ["</search>\n", "</search>", "</search>\n\n", "</answer>", "</answer>\n", "</answer>\n\n", "></search"]
        
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # prompts = [self.prompt_template.get_string(question=question) for question in dataset.question]
        messagess = [[{'role':'system','content':self.system_prompt},{'role':'user','content':self.user_prompt.format(question=question)}] for question in dataset.question]
        prompts = [self.prompt_template.get_string(messages=messages) for messages in messagess]
        dataset.update_output('messages',messagess)
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
                if self.end_of_answer_token in step_output and ((self.end_of_answer_token in step_output) or step_output.endswith("<|endoftext|>")):
                    item.pred = str(extract_between(step_output, self.begin_of_answer_token, self.end_of_answer_token))
                    item.finish_flag = True
                    item.finish_reason = "Finished"
                
                elif self.begin_of_search_token in step_output:
                    if "</search>" not in step_output:
                        step_output += '>'
                    queries = extract_between(step_output, self.begin_of_search_token, self.end_of_search_token)
                    if queries:
                        queries = extract_between_all(queries, self.begin_of_query_token, self.end_of_query_token)                    
                    if queries is not None:
                        item.messages.append({'role':'user','content':step_output.strip()})
                        item.prompt = self.prompt_template.get_string(messages=item.messages)
                        step_query_list.append({'item': item, 'queries': queries, 'item_query_num': len(queries)})
                    else:
                        item.pred = 'No valid answer found'
                        item.finish_flag = True
                        item.finish_reason = 'Query instruction error'

                else:
                    item.pred = step_output.strip()
                    item.messages.append({'role':'user','content':self.error_prompt})
                    item.prompt = self.prompt_template.get_string(messages=item.messages)
                    # item.finish_flag = False
                    # item.finish_reason = 'Normal finish without answer pattern'
                
            # do retrieval and add retrieved docs to prompt
            if len(step_query_list) > 0:
                # print(dataset.question)
                _retrieved_docs = self.retriever.batch_search(sum([it['queries'] for it in step_query_list],[]))
                retrieved_docs = []
                doc_index = 0
                for it in step_query_list:
                    item=it['item']
                    item_query_num = it['item_query_num']
                    item_retrieved_docs = sum(_retrieved_docs[doc_index:doc_index+item_query_num],[])
                    doc_index += item_query_num
                    retrieved_docs.append(item_retrieved_docs)
                for it, item_retrieved_docs in zip(step_query_list, retrieved_docs):
                    item = it['item']
                    queries = it['queries']
                    item.retrieval_results[item.retrieved_times] = {'queries': queries,'docs': copy.copy(item_retrieved_docs)}
                    # print(item_retrieved_docs)
                    learning_str = self._retrieved_docs_to_string(item_retrieved_docs)
                    extra_info = self.extra_prompt.format(learning_str=learning_str)
                    item.messages.append({'role':'user','content':extra_info})
                    item.prompt = self.prompt_template.get_string(messages=item.messages)
                    item.retrieved_times += 1

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset


class ReaRAGPipeline(BasicPipeline):
    rearag_system_prompt = rearag_system_prompt = '''Your task is to solve a question answering task. To improve your solving accuracy, please conduct reasoning process interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action are in the form of function, there are two types:

# Available functions:
(1) search
{
    "name": "search",
    "description": "It can help you find useful information through the internet or local knowledge base. You can use this tool to access external knowledge",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "description": "what you want to search"
            }
        },
        "required": [
            "query"
        ]
    }
}

(2) finish
{
    "name": "finish",
    "description": "You can use this function to make a conclusion from the reasoning process and give the final answer. The reasoining process is completed after this `finish` function is called",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "description": "the final answer"
            }
        },
        "required": [
            "answer"
        ]
    }
}

Please follow the format strictly.
'''
    user_prompt = '''{question}'''
    long_ans_prompt = "If the Question is comparison type, do not refer the given Context, else, answer the Question based on the given Contexts.\n\nContext: {}\n\nQuestion: {}\n\nAnswer:"
    extract_short_ans_prompt = '''The Reference Answer is the final answer to the question. It's the final deterministic answer, your task is to give concise version of it. Only give me the short answer and do not output any other words.
[Question]
{question}
[Reference answer]
{reference_ans}

Only give me the short answer and do not output any other words. For yes, or no answer, only answer it short. Give the shortest answer possible.
'''

    def __init__(self,config,
        prompt_template=None,
        max_iter_num=15,
        begin_of_function_token="```",
        end_of_function_token="```",
        retriever=None, 
        generator=None,):
        if prompt_template is None:
            prompt_template = PromptTemplate(
                config=config,
                system_prompt=self.rearag_system_prompt,
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
        self.max_iter_num = max_iter_num
        self.begin_of_function_token = begin_of_function_token
        self.end_of_function_token = end_of_function_token
        self.AgentUtils = AgentUtils()
        self.sample_generate_params = {
            'temperature': 0,
            'top_p': 0.85,
            'stop': ["<|user|>", "<|observation|>", "<|assistant|>"],
        }
        self.rag_generate_params = {
            'top_p': 0.7,
            'temperature': 0,
            'stop': ["<|user|>", "<|endoftext|>", "<|assistant|>"]
        }
        
    def template_doc_and_tempquery(self, retrieved_docs: List[Dict], question:str):
        chunks = []
        for _ in retrieved_docs:
            if _ not in chunks:
                chunks.append(_['contents'])
        prompt = self.long_ans_prompt.format('\n\n'.join(chunks), question)
        return prompt
            
            
    def small_batch_run(self, dataset, do_eval=True, pred_process_fun=None):
        dataset.messages = [[{"role": "system", "content": self.rearag_system_prompt},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": ''},] for question in dataset.question]
        prompts = [self.prompt_template.get_string(messages=messages) for messages in dataset.messages]
        dataset.update_output('messages',dataset.messages)
        dataset.update_output('prompt',prompts)
        dataset.update_output('finish_flag',[False]*len(prompts))
        dataset.update_output('status',['start']*len(prompts))
        dataset.update_output('retrieval_results',[{} for _ in range(len(prompts))])
        dataset.update_output('retrieved_times',[0]*len(prompts))
        
        for current_step_idx in range(self.max_iter_num + 1):
            exist_items = [item for item in dataset if item.finish_flag == False]
            exist_prompts = [item.prompt for item in exist_items]
            print(f"Current step: {current_step_idx}, exist_items: {len(exist_items)}")

            if len(exist_items) == 0:
                print("All prompts are finished")
                break
            if current_step_idx == self.max_iter_num:
                print("Max retrieval number reached")
                for item in exist_items:
                    item.pred = "No valid answer found"
                    item.finish_flag = True
                    item.finish_reason = "Reach max retrieval number"
                break
            
            step_outputs = self.generator.generate(exist_prompts,**self.sample_generate_params)
            # print(step_outputs)           
            
            step_query_list = []
            
            for item, step_output in zip(exist_items,step_outputs):
                try:
                    thoughts, actions = self.AgentUtils.postprocess_agent_response(step_output)
                except Exception as e:
                    print(f"Error in postprocess_agent_response: {e}")
                    item.status = "repeat"
                    continue
                try:
                    thought, action = thoughts[0], actions[0]
                except:
                    item.status = "repeat"
                    continue
                action_type = action["function"]
                if action_type not in ["search","finish"]:
                    item.status = "repeat"
                    continue
                if action_type == "finish":
                    reference_ans = action["parameters"]["answer"]
                    prompt = self.extract_short_ans_prompt.format(question=item.question,reference_ans=reference_ans)
                    final_message = [{"role":"user", "content":prompt}]
                    prompt = self.prompt_template.get_string(messages=final_message)
                    final_answer = self.generator.generate(prompt,**self.rag_generate_params)[0]
                    item.finish_flag = True
                    item.pred = final_answer
                    item.finish_reason = "Finished"
                if action_type == "search":
                    item.status = "search"
                    query = action["parameters"]["query"]
                    step_query_list.append({'item': item, 'query': query})
                    item.messages[-1] = {"role": "assistant", "content": step_output}
            
            # 处理repeat状态的items
            repeat_items = [item for item in exist_items if item.status == "repeat"]
            for item in repeat_items:
                # 重置消息到初始状态
                item.messages = [{"role": "system", "content": self.rearag_system_prompt},
                               {"role": "user", "content": item.question},
                               {"role": "assistant", "content": ''}]
                item.prompt = self.prompt_template.get_string(messages=item.messages)
                item.status = "repeat"

            if len(step_query_list) > 0:
                retrieved_docs = self.retriever.batch_search([it['query'] for it in step_query_list])
                temp_prompts = []
                for it, item_retrieved_docs in zip(step_query_list, retrieved_docs):
                    item = it['item']
                    query = it['query']
                    item.retrieval_results[item.retrieved_times] = {'query': query,'docs': copy.copy(item_retrieved_docs)}
                    #item.retrieved_docs += [item_retrieved_docs]
                    temp_prompt = self.template_doc_and_tempquery(item_retrieved_docs,query)
                    temp_prompts.append(temp_prompt)
                    item.retrieved_times += 1
                
                observations = self.generator.generate(temp_prompts)
                
                for observation, it in zip(observations, step_query_list):
                    item = it['item']
                    item.messages.append({"role": "observation", "content": observation})
                    item.messages.append({"role": "assistant", "content": ''})
                    item.prompt = self.prompt_template.get_string(messages=item.messages)
        
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset
                    
    def run(self, dataset, do_eval=True, pred_process_fun=None, batch_size=128):
        """按批次处理数据集
        
        Args:
            dataset: 输入数据集
            do_eval: 是否进行评估
            pred_process_fun: 预测结果处理函数
            batch_size: 批处理大小
        
        Returns:
            处理后的完整数据集
        """
        processed_datasets = []
        
        # 按批次处理数据
        for batch_dataset in get_batch_dataset(dataset,batch_size):
            processed_batch = self.small_batch_run(
                batch_dataset, 
                do_eval=False,  # 先不进行评估
                pred_process_fun=pred_process_fun
            )
            processed_datasets.append(processed_batch)
        
        # 合并所有处理后的批次
        merged_dataset = merge_batch_dataset(processed_datasets)
        
        # 在合并后的完整数据集上进行评估
        if do_eval:
            merged_dataset = self.evaluate(merged_dataset, pred_process_fun=pred_process_fun)
            
        return merged_dataset
                    

            
class CoRAGPipeline(BasicPipeline):
    extract_short_ans_prompt = '''The Reference Answer is the final answer to the question. It's the final deterministic answer, your task is to give concise version of it. Only give me the short answer and do not output any other words.
[Question]
{question}
[Reference answer]
{reference_ans}

Only give me the short answer and do not output any other words. For yes, or no answer, only answer it short. Give the shortest answer possible.
'''
    def __init__(self,config,
                 max_iter_num=4, 
                 prompt_template=None,
                 retriever=None, 
                 generator=None,):
        super().__init__(config, prompt_template)
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator
        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever
        self.task_desc = config['task_desc']
        self.max_iter_num = max_iter_num
        self.generate_subquery_params = {
            'temperature': 0,
            'do_sample': False,
            'max_tokens': 64,
            # 'n':15,
        }
        self.generate_intermediate_answer_params = {
            'temperature': 0,
            'do_sample': False,
            'max_tokens': 64,
        }
        self.generate_final_answer_params = {
            'temperature': 0,
            'do_sample': False,
            'max_tokens': 64,
            'skip_special_tokens': True,
        }

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        question = dataset.question
        dataset.update_output('sub_queries',[[] for _ in range(len(question))])
        dataset.update_output('sub_answers',[[] for _ in range(len(question))])
        dataset.update_output('sub_retrieval_results',[{} for _ in range(len(question))])

        dataset.update_output('subquery_messages',[[] for _ in range(len(question))])
        dataset.update_output('intermediate_answer_messages',[[] for _ in range(len(question))])
        dataset.update_output('final_answer_messages',[[] for _ in range(len(question))])

        dataset.update_output('subquery_prompts',[[] for _ in range(len(question))])
        dataset.update_output('intermediate_answer_prompts',[[] for _ in range(len(question))])
        dataset.update_output('final_answer_prompts',[[] for _ in range(len(question))])
        dataset.update_output('finish_flag',[False]*len(question))
        # 最大迭代次数

        for now_iter_num in range(self.max_iter_num):
            # 先对于每个question，生成一个subquery
            now_items = [item for item in dataset if item.finish_flag == False]
            subquery_messages =  [get_generate_subquery_message(item.question, item.sub_queries, item.sub_answers, self.task_desc) for item in now_items]

            # 先使用generator生成subquery
            for item, subquery_message in zip(now_items, subquery_messages):
                item.subquery_messages.append(subquery_message)
                item.subquery_prompts.append(self.prompt_template.get_string(messages=item.subquery_messages[-1]))
            subquery_responses = self.generator.generate([item.subquery_prompts[-1] for item in now_items],**self.generate_subquery_params)
            for item, subquery_response in zip(now_items, subquery_responses):
                item.sub_queries.append(self._normalize_subquery(subquery_response))

            temp_retrieval_results = self.retriever.batch_search([item.sub_queries[-1] for item in now_items])

            for item, temp_retrieval_result in zip(now_items, temp_retrieval_results):
                item.sub_retrieval_results[now_iter_num] = temp_retrieval_result

            intermediate_answer_messages = [get_generate_intermediate_answer_message(item.sub_queries[-1], item.sub_retrieval_results[now_iter_num]) for item in now_items]
            intermediate_answer_prompts = [self.prompt_template.get_string(messages=intermediate_answer_message) for intermediate_answer_message in intermediate_answer_messages]

            for item, intermediate_answer_message, intermediate_answer_prompt in zip(now_items, intermediate_answer_messages, intermediate_answer_prompts):
                item.intermediate_answer_messages.append(intermediate_answer_message)
                item.intermediate_answer_prompts.append(intermediate_answer_prompt)
            
            intermediate_answer_responses = self.generator.generate(intermediate_answer_prompts,**self.generate_intermediate_answer_params)
            for item, intermediate_answer_response in zip(now_items, intermediate_answer_responses):
                item.sub_answers.append(intermediate_answer_response)

        # 使用generator生成final_answer
        final_answer_messages = [get_generate_final_answer_message(item.question, item.sub_queries, item.sub_answers,self.task_desc,sum(list(item.sub_retrieval_results.values()),[])) for item in dataset]
        final_answer_prompts = [self.prompt_template.get_string(messages=final_answer_message) for final_answer_message in final_answer_messages]

        final_answer_responses = self.generator.generate(final_answer_prompts,**self.generate_final_answer_params)
        dataset.update_output('pred',final_answer_responses)


        if do_eval:
            dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset
        
    def _normalize_subquery(self,subquery: str) -> str:
        subquery = subquery.strip()
        if subquery.startswith('"') and subquery.endswith('"'):
            subquery = subquery[1:-1]
        if subquery.startswith('Intermediate query'):
            subquery = re.sub(r'^Intermediate query \d+: ', '', subquery)
        return subquery
