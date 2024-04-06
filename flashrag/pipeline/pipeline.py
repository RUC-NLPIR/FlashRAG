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

    def build_prompt(self, dataset, prompt_templete=None, use_reference = True, reference = None):
        base_templete_rag = 'Write a high-quality answer for the given question using the provided information (some of which might be irrelevant).\n{reference}\nQuestion:{question}\nAnswer:'
        base_templete_standard = 'Write a high-quality answer for the given question.\nQuestion:{question}\nAnswer:'
        
        if prompt_templete is None:
            if use_reference:
                prompt_templete = base_templete_rag
            else:
                prompt_templete = base_templete_standard
        prompt_list = []    

        if reference is not None:
            assert len(reference) == len(dataset)
            
        for idx,item in enumerate(dataset):
            if use_reference:
                if reference is not None:
                    # use provided reference
                    format_reference = reference[idx]
                else:
                    format_reference = ''
                    for idx, doc_item in enumerate(item.retrieval_result):
                        content = doc_item['contents']
                        title = content.split("\n")[0]
                        text = "\n".join(content.split("\n")[1:])
                        format_reference += f"{idx+1}(Title: {title}) {text}\n"

                prompt = prompt_templete.format(question = item.question, reference = format_reference)
            else:
                prompt = prompt_templete.format(question = item.question)
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
        
        if config['refiner_name'] is not None:
            self.refiner = get_refiner(config)
        else:
            self.refiner = None
    
    def standard_run(self, dataset):
        # direct generation without RAG
        input_prompts = self.build_prompt(dataset, use_reference=False)
        dataset.update_output('prompt', input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred",pred_answer_list)

        return dataset

    def run(self, dataset, do_eval=False):
        input_query = dataset.question
        if self.rewriter:
            input_query = self.rewriter.batch_run(input_query)
            dataset.update_output('rewrite_query', input_query)
  
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output('retrieval_result', retrieval_results)

        if self.refiner:
            if 'llmlingua' in self.refiner.name:
                # input prompt
                input_prompts = self.build_prompt(dataset)
                dataset.update_output('prompt', input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
                dataset.update_output('prompt', input_prompts)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output('refine_result', refine_results)
                input_prompts = self.build_prompt(dataset, reference=refine_results)
        else:
            input_prompts = self.build_prompt(dataset)
            dataset.update_output('prompt', input_prompts)
    
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred",pred_answer_list)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

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
    
    def run(self, dataset, do_eval=False):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)

        # split dataset based on judge_result
        pos_dataset, neg_dataset = split_dataset(dataset, judge_result)

        pos_dataset = self.sequential_pipeline.run(pos_dataset)
        neg_dataset = self.sequential_pipeline.standard_run(neg_dataset)

        # merge datasets into original format
        dataset = merge_dataset(pos_dataset, neg_dataset, judge_result)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        return dataset




