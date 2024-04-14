from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger


class BasicPipeline:
    r"""Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    
    """
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.evaluator = Evaluator(config)

    def run(self, dataset):
        r"""The overall inference process of a RAG framework.
        
        """
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        r"""The evaluation process after finishing overall generation"""
        if pred_process_fun is not None:
            raw_pred = dataset.pred
            processed_pred = [pred_process_fun(pred) for pred in raw_pred]
            dataset.update_output('raw_pred',raw_pred)
            dataset.update_output('pred', processed_pred)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        return dataset
        
    def format_reference(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

    def build_prompt(self, question_list, 
                     retrieval_results=None,
                     prompt_templete=None, 
                     use_reference = True, 
                     reference = None,
                     previous_gen = ""):
        
        base_templete_rag = "[INST] <<SYS>> Answer the question based on the given document. Only give me the answer and do not output any other words.\n\nThe following are given document.\n{reference}\nAnswer the question based on the given information. Only give me the answer and do not output any other words.<</SYS>>\nQuestion: {question}\nAnswer:[/INST]{previous_gen}"
        base_templete_standard = "[INST] <<SYS>> Answer the question based on your own knowledge. Only give me the answer and do not output any other words.<</SYS>>\nQuestion: {question}\nAnswer:[/INST]{previous_gen}"

        #base_templete_rag = "Answer the question based on the given information. Only give me the answer and do not output any other words.\n\nThe following are given information.\n{reference}\n\nAnswer the question based on the given information. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:{previous_gen}"
        #base_templete_standard = "Answer the question based on your own knowledge. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:{previous_gen}"
        #base_templete_rag = 'Write a high-quality answer for the given question using the provided information (some of which might be irrelevant).\n{reference}\nQuestion:{question}\nAnswer:{previous_gen}'
        #base_templete_standard = 'Write a high-quality answer for the given question.\nQuestion:{question}\nAnswer:{previous_gen}'
        
        if prompt_templete is None:
            if use_reference:
                prompt_templete = base_templete_rag
            else:
                prompt_templete = base_templete_standard
        prompt_list = []    

        if reference is not None:
            assert len(reference) == len(question_list)

        for idx in range(len(question_list)):
            question = question_list[idx]
            if use_reference:
                if reference is not None:
                    # use provided reference
                    format_reference = reference[idx]
                else:
                    retrieval_result = retrieval_results[idx]
                    format_reference = self.format_reference(retrieval_result)
                
                prompt = prompt_templete.format(
                    question = question, reference = format_reference, previous_gen = previous_gen)
            else:
                prompt = prompt_templete.format(question = question, previous_gen = previous_gen)
            prompt_list.append(prompt)

        return prompt_list

    
class SequentialPipeline(BasicPipeline):
    def __init__(self, config):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """
        super().__init__(config)
        self.retriever = get_retriever(config)
        self.generator = get_generator(config)
        if config['rewriter_path'] is not None:
            self.rewriter = get_rewriter(config)
        else:
            self.rewriter = None
        
        self.use_fid = config['use_fid']

        if config['refiner_name'] is not None:
            self.refiner = get_refiner(config)
        else:
            self.refiner = None
    
    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = self.build_prompt(dataset.question, use_reference=False)
        dataset.update_output('prompt', input_prompts)
        if self.use_fid:
            print('Use FiD generation')
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append(
                    [q + " " + doc for doc in docs]
                )

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred",pred_answer_list)
        
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question
        if self.rewriter:
            input_query = self.rewriter.batch_run(input_query)
            dataset.update_output('rewrite_query', input_query)
  
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output('retrieval_result', retrieval_results)

        if self.refiner:
            if 'llmlingua' in self.refiner.name:
                # input prompt
                input_prompts = self.build_prompt(dataset.question, dataset.retrieval_result)
                dataset.update_output('prompt', input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
                dataset.update_output('prompt', input_prompts)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output('refine_result', refine_results)
                input_prompts = self.build_prompt(dataset.question, dataset.retrieval_result, reference=refine_results)
        else:
            input_prompts = self.build_prompt(dataset.question, dataset.retrieval_result)
            dataset.update_output('prompt', input_prompts)

        if self.use_fid:
            print('Use FiD generation')
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append(
                    [q + " " + doc for doc in docs]
                )
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred",pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset

class ConditionalPipeline(BasicPipeline):
    def __init__(self, config):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """
        super().__init__(config)
        self.judger = get_judger(config)

        self.sequential_pipeline = SequentialPipeline(config)
    
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)

        # split dataset based on judge_result
        pos_dataset, neg_dataset = split_dataset(dataset, judge_result)

        pos_dataset = self.sequential_pipeline.run(pos_dataset)
        neg_dataset = self.sequential_pipeline.standard_run(neg_dataset)

        # merge datasets into original format
        dataset = merge_dataset(pos_dataset, neg_dataset, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        
        return dataset




