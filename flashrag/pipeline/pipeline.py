from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator, get_refiner


class BasicPipeline:
    r"""Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    
    """
    def __init__(self, config):
        self.config = config
        self.evaluator = Evaluator(config)

    def run(self, dataset):
        r"""The overall inference process of a RAG framework.
        
        """
        pass

    def build_prompt(self, dataset, prompt_templete=None, reference = None):
        base_templete = 'Write a high-quality answer for the given question using the provided information (some of which might be irrelevant).\n{reference}\nQuestion:{question}\nAnswer:'
        if prompt_templete is None:
            prompt_templete = base_templete
        prompt_list = []    

        if reference is not None:
            assert len(reference) == len(dataset)
            
        for idx,item in enumerate(dataset):
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
            prompt_list.append(prompt)
        

    
class SequentialPipeline(BasicPipeline):
    def __init__(self, config):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """
        super().__init__(config)
        self.retriever = get_retriever(config)
        if config['rewriter_path'] is not None:
            self.rewriter = get_rewriter(config)
        else:
            self.rewriter = None
        
        if config['refiner_name'] is not None:
            self.refiner = get_refiner(config)
        else:
            self.refiner = None
    
    def run(self, dataset):
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

