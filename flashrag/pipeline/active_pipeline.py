import transformers
import torch
from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator
from flashrag.pipeline import BasicPipeline


class ITERRETGENPipeline(BasicPipeline):
    def __init__(self, config, iter_num = 3):
        super().__init__(config)
        self.iter_num = iter_num
        self.retriever = get_retriever(config)
        self.generator = get_generator(config)
    
    def run(self, dataset, do_eval=False, pred_process_fun=None):
        questions = dataset.question

        # run in batch
        past_generation_result = [] # list of N items
        for iter_idx in range(self.iter_num):
            if iter_idx == 0:
                input_query = questions
            else:
                assert len(questions) == len(past_generation_result)
                input_query = [f"{q} {r}" for q,r in zip(questions, past_generation_result)]
            
            # generation-augmented retrieval
            retrieval_results = self.retriever.batch_search(input_query)
            dataset.update_output(f'retrieval_result_iter_{iter_idx}', retrieval_results)
            
            # retrieval-augmented generation
            input_prompts = self.build_prompt(questions, retrieval_results)
            dataset.update_output(f'prompt_iter_{iter_idx}', input_prompts)
            past_generation_result = self.generator.generate(input_prompts)
            dataset.update_output(f'pred_iter_{iter_idx}', past_generation_result)
        
        dataset.update_output("pred", past_generation_result)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset






