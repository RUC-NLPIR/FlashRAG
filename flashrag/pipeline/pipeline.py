from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator


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
        
        if config['refiner_path'] is not None:
            self.refiner = get_refiner(config)
        else:
            self.refiner = None
    
    def run(self, dataset):
        input_query = dataset.question
        if self.rewriter:
            input_query = self.rewriter.batch_run(input_query)
            dataset.update('rewrite_query', input_query)
  
        # TODO: batch search
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update('retrieval_result', retrieval_results)

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
        if self.refiner:
            # TODO: fix more items into input
            input_prompts = self.refiner.batch_run(input_prompts)
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred",pred_answer_list)


class NaiveRAG(BasicPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.retriever = get_retriever(config)
        self.generator = get_generator(config)
        self.evaluator = Evaluator(config)
    
    def run(self, dataset):
        r"""Run the overall inference process of a naive RAG process.
        Retrieve each query directly, construct a prompt, and generate answers.
        
        """

        # Retrieval
        # TODO: batch retrieval

        for item in dataset:
            retrieval_result = self.retriever.search(item.question)
            item.update_output("retrieval_result", retrieval_result)

        
        # Build prompt
        # TODO: modify into class
        prompt_templete = 'Write a high-quality answer for the given question using the provided information (some of which might be irrelevant).\n\n{reference}\n\nQuestion:{question}\nAnswer:'
        format_reference = ''
        for item in dataset:
            for idx, doc_item in enumerate(item.retrieval_result):
                content = doc_item['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"{idx+1}(Title: {title}) {text}\n"
            prompt = prompt_templete.format(question = item.question, reference = format_reference)
            item.update_output("prompt", prompt)
        
        # generate
        input_list = dataset.get_attr_data("prompt")
        pred_answer_list = self.generator.generate(input_list)
        
        dataset.update_output("pred",pred_answer_list)
        # evaluate & save result
        eval_result = self.evaluator.evaluate(dataset)
        return dataset, eval_result