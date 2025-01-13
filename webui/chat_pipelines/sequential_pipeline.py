from flashrag.pipeline import SequentialPipeline
from flashrag.dataset import Dataset, Item
from chat_pipelines.base_chat_pipeline import BaseChatPipeline

"""
Insert yield into the chat process.
Add `chat` function.
"""

class SequentialPipeline_Chat(BaseChatPipeline, SequentialPipeline):
    def __init__(self, *args, **kwargs):
        SequentialPipeline.__init__(self, *args, **kwargs)
        config = self.config
        BaseChatPipeline.__init__(self, config)

    def chat(self, query):
        dataset = Dataset(
            config = self.config, data = [Item({"question": query})]
        )

        input_query = dataset.question
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)

        yield from self.display_retrieval_result(retrieval_results[0])

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
                yield from self.display_middle_result(input_prompts, 'Refining Prompts')
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]
                yield from self.display_middle_result(input_prompts, 'Refining Documents')

        else:
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(dataset.question, dataset.retrieval_result)
            ]
        dataset.update_output("prompt", input_prompts)

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc for doc in docs])
        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        yield from self.display_middle_result(pred_answer_list, 'Final Answer')
    

class NaivePipeline_Chat(BaseChatPipeline, SequentialPipeline):
    def __init__(self, *args, **kwargs):
        SequentialPipeline.__init__(self, *args, **kwargs)
        config = self.config
        BaseChatPipeline.__init__(self, config)

    def chat(self, query):
        dataset = Dataset(config = self.config, data = [Item({"question": query})])

        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", input_prompts)

        yield from self.display_middle_result(input_prompts, 'Input prompt')

        pred_answer_list = self.generator.generate(input_prompts)
        
        yield from self.display_middle_result(pred_answer_list, 'Final Answer')
