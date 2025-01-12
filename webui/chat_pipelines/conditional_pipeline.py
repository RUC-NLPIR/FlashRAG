from flashrag.pipeline import SequentialPipeline, ConditionalPipeline, AdaptivePipeline
from flashrag.dataset import Dataset, Item
from flashrag.utils import get_judger
from flashrag.prompt import PromptTemplate
from chat_pipelines.base_chat_pipeline import BaseChatPipeline
from chat_pipelines.sequential_pipeline import SequentialPipeline_Chat, NaivePipeline_Chat
from chat_pipelines.active_pipeline import IRCOTPipeline_Chat


class ConditionalPipeline_Chat(BaseChatPipeline, ConditionalPipeline):
    def __init__(
            self, 
            config, 
            prompt_template = None,
            retriever = None,
            generator = None
        ):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """
        BaseChatPipeline.__init__(self, config)
        self.config = config
        self.retriever = retriever
        self.generator = generator
        self.judger = get_judger(config)
        self.sequential_pipeline = SequentialPipeline_Chat(
            config, prompt_template, retriever=self.retriever, generator=self.generator
        )

        self.zero_shot_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}",
        )
        self.zero_shot_pipeline = NaivePipeline_Chat(
            config, self.zero_shot_templete, retriever=self.retriever, generator=self.generator
        )

    def chat(self, query):
        dataset = Dataset(config = self.config, data = [Item({"question": query})])

        judge_result = self.judger.judge(dataset)[0]
        judge_result_str = "need retrieval" if judge_result else "no need retrieval"
        yield "<strong>Judge result:\n</strong>" + judge_result_str

        if judge_result:
            # use retrieval
            yield from self.sequential_pipeline.chat(query)
        else:
            # use zero-shot
            yield from self.zero_shot_pipeline.chat(query)


class AdaptivePipeline_Chat(BaseChatPipeline, AdaptivePipeline):
    def __init__(
            self, 
            config, 
            prompt_template = None,
            retriever = None,
            generator = None
    ):
        BaseChatPipeline.__init__(self, config)
        self.config = config
        self.judger = get_judger(config)
        self.generator = generator
        self.retriever = retriever

        # Load three pipeline for three types of query: naive/single-hop/multi-hop
        norag_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}",
        )
        self.norag_pipeline = NaivePipeline_Chat(
            config,
            prompt_template =norag_templete,
            retriever=retriever,
            generator=generator,
        )

        self.single_hop_pipeline = SequentialPipeline_Chat(
            config,
            prompt_template=prompt_template,
            retriever=retriever,
            generator=generator,
        )
        
        self.multi_hop_pipeline = IRCOTPipeline_Chat(
            config,
            retriever=retriever,
            generator=generator
        )
    
    def chat(self, query):
        dataset = Dataset(config = self.config, data = [Item({"question": query})])

        judge_result = self.judger.judge(dataset)[0]
        if judge_result == 'A':
            judge_result_str = 'no need retrieval'
            yield "<strong>Judge result:\n</strong>" + judge_result_str
            yield from self.norag_pipeline.chat(query)
        elif judge_result == 'B':
            judge_result_str = 'single-hop'
            yield "<strong>Judge result:\n</strong>" + judge_result_str
            yield from self.single_hop_pipeline.chat(query)
        elif judge_result == 'C':
            judge_result_str = 'multi-hop'
            yield "<strong>Judge result:\n</strong>" + judge_result_str
            yield from self.multi_hop_pipeline.chat(query)
