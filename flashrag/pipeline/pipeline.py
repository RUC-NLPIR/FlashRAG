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
            retrieval_results = self.refiner.batch_run(dataset)
            dataset.update_output('retrieval_result', retrieval_results)

        prompt_templete = 'Write a high-quality answer for the given question using the provided information (some of which might be irrelevant).\n\n{reference}\n\nQuestion:{question}\nAnswer:'
        # TODO: batch prompt building
        format_reference = ''
        for item in dataset:
            for idx, doc_item in enumerate(item.retrieval_result):
                content = doc_item['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"{idx+1}(Title: {title}) {text}\n"
            prompt = prompt_templete.format(question = item.question, reference = format_reference)
            item.update_output("prompt", prompt)
        
        input_prompts = dataset.get_attr_data("prompt")
    
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred",pred_answer_list)

