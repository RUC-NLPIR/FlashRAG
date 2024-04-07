import transformers
from transformers import LogitsProcessorList
import torch
from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator
from flashrag.pipeline import BasicPipeline
from flashrag.pipeline.replug_utils import REPLUGLogitsProcessor, load_replug_model


class REPLUGPipeline(BasicPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.retriever = get_retriever(config)
        # load specify model for REPLUG
        model = load_replug_model(config['generator_model_path'])
        self.generator = get_generator(config, model=model)
    


    def build_single_doc_prompt(self, question, doc_list, prompt_templete=None):
        base_templete_rag = 'Write a high-quality answer for the given question using the provided information (some of which might be irrelevant).\n{reference}\nQuestion:{question}\nAnswer:'
        if prompt_templete is None:
            prompt_templete = base_templete_rag

        return [prompt_templete.format(reference = doc, question = question) for doc in doc_list]


    def run(self, dataset, do_eval=False, pred_process_fun=None):
        input_query = dataset.question

        retrieval_results, doc_scores = self.retriever.batch_search(input_query, return_score=True)
        dataset.update_output('retrieval_result', retrieval_results)
        dataset.update_output('doc_scores', doc_scores)

        pred_answer_list = []
        # each doc has a prompt
        for item in dataset:
            docs = [doc_item['contents'] for doc_item in item.retrieval_result]
            prompts = self.build_single_doc_prompt(question=item.question, doc_list=docs)
            scores = torch.tensor(item.doc_scores, dtype=torch.float32).to(self.device)
            output = self.generator.generate(prompts, 
                                    batch_size=len(docs), 
                                    logits_processor = LogitsProcessorList([REPLUGLogitsProcessor(scores)])
                                )
            # the output of the batch is same
            output = output[0]
            pred_answer_list.append(output)
        
        dataset.update_output("pred",pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset