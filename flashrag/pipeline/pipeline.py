from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate
import torch
import copy
from torch.multiprocessing import Pool
import os
from flashrag.dataset.dataset import Dataset
from tqdm import tqdm


class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config["save_retrieval_cache"]
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            dataset = pred_process_fun(dataset)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        # save retrieval cache
        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        # TODO: add rewriter module

        self.use_fid = config["use_fid"]

        if config["refiner_name"] is not None:
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = [
            self.prompt_template.get_string(question=q) for q in dataset.question
        ]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(
            dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
        )
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)

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
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            if not self.use_fid:
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc['contents'] for doc in docs])
        dataset.update_output("prompt", input_prompts)

        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(
            dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
        )

        return dataset

    def run_with_refiner_on_multicard(
        self, dataset, do_eval=True, pred_process_fun=None
    ):
        input_query = dataset.question

        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)

        if self.refiner:
            if hasattr(self.refiner, 'refiner'):
                del self.refiner.refiner
            del self.refiner
            torch.cuda.empty_cache()
            torch.multiprocessing.set_start_method("spawn", force=True)
            # 获取GPU列表
            gpu_ids = [int(id.strip()) for id in self.config["gpu_id"].split(",")]
            num_gpus = len(gpu_ids)
            # 将数据集平均分割
            data_chunks = split_list(dataset.data, num_gpus)
            # 准备参数列表
            args_list = []
            for i in range(num_gpus):
                chunk_data = data_chunks[i]
                gpu_id = gpu_ids[i]
                args = (chunk_data, gpu_id, self.config)
                args_list.append(args)
            # 多进程并行运行refiner
            with Pool(processes=num_gpus) as pool:
                results = pool.map(process_refiner_chunk, args_list)
            # 合并结果，保持顺序
            refine_results = []
            for chunk_result in results:
                refine_results.extend(chunk_result)
            dataset.update_output("refine_result", refine_results)
            # 生成输入提示
            input_prompts = [
                self.prompt_template.get_string(question=q, formatted_reference=r)
                for q, r in zip(dataset.question, refine_results)
            ]
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
        # 删除refiner以释放内存
        if self.config["refiner_name"] is not None:
            if "kg" in self.config["refiner_name"].lower():
                self.refiner = get_refiner(self.config, self.retriever, self.generator)
                self.generator = self.refiner.generator
            else:
                self.generator = get_generator(self.config)

        # 生成答案
        pred_answer_list_final = []
        score_answer_list_final = []
        for i in tqdm(range(self.generation_count)):
            pred_answer_list = self.generator.generate(
                input_prompts, return_scores=True
            )
            pred_answer_list_final.append(pred_answer_list[0])
            score_answer_list_final.append(pred_answer_list[1])
            if i == 0:
                dataset.update_output("pred", pred_answer_list[0])
        final_preds = [[] for _ in range(len(dataset.data))]
        final_scores = [[] for _ in range(len(dataset.data))]
        for generated_set in pred_answer_list_final:
            for i in range(len(dataset.data)):
                final_preds[i].append(generated_set[i])
        for generated_set in score_answer_list_final:
            for i in range(len(dataset.data)):
                final_scores[i].append(generated_set[i])
        dataset.update_output("preds", final_preds)
        dataset.update_output("scores", final_scores)
        dataset = self.evaluate(
            dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
        )

        return dataset


def split_list(lst, n):
    """Split list `lst` into `n` approximately equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def process_refiner_chunk(args):
    chunk_data, gpu_id, config = args
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)  # 在每个进程中，设备索引从0开始
    # 更新配置
    config_copy = copy.deepcopy(config)
    config_copy["device"] = torch.device("cuda:0")
    # 创建refiner
    refiner = get_refiner(config_copy)
    # 创建子数据集
    sub_dataset = Dataset(config=config_copy, data=chunk_data)
    # 运行refiner
    refine_results = refiner.batch_run(sub_dataset)
    # 释放refiner资源
    del refiner
    torch.cuda.empty_cache()
    return refine_results

class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)

        self.judger = get_judger(config)
        if generator is None:
            self.generator = get_generator(config)
        if retriever is None:
            self.retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever

        self.sequential_pipeline = SequentialPipeline(
            config, prompt_template, retriever=self.retriever, generator=self.generator
        )

        self.zero_shot_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}",
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        pos_dataset, neg_dataset = dataset_split[True], dataset_split[False]

        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(
            dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
        )

        return dataset


class AdaptivePipeline(BasicPipeline):
    def __init__(
        self,
        config,
        norag_template=None,
        single_hop_prompt_template=None,
        multi_hop_prompt_template=None,
        retriever = None,
        generator = None
    ):
        super().__init__(config)
        # load adaptive classifier as judger
        self.judger = get_judger(config)

        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)

        self.generator = generator
        self.retriever = retriever

        # Load three pipeline for three types of query: naive/single-hop/multi-hop
        from flashrag.pipeline import IRCOTPipeline

        if norag_template is None:
            norag_templete = PromptTemplate(
                config=config,
                system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
                user_prompt="Question: {question}",
            )
        self.norag_pipeline = SequentialPipeline(
            config,
            prompt_template=norag_templete,
            retriever=retriever,
            generator=generator,
        )

        self.single_hop_pipeline = SequentialPipeline(
            config,
            prompt_template=single_hop_prompt_template,
            retriever=retriever,
            generator=generator,
        )

        self.multi_hop_pipeline = IRCOTPipeline(
            config, prompt_template=multi_hop_prompt_template, retriever=retriever, generator=generator, max_iter=5
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: choice result representing which pipeline to use(e.g. A, B, C)
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        for symbol, symbol_dataset in dataset_split.items():
            if symbol == "A":
                symbol_dataset = self.norag_pipeline.naive_run(
                    symbol_dataset, do_eval=False
                )
            elif symbol == "B":
                symbol_dataset = self.single_hop_pipeline.run(
                    symbol_dataset, do_eval=False
                )
            elif symbol == "C":
                symbol_dataset = self.multi_hop_pipeline.run(
                    symbol_dataset, do_eval=False
                )
            else:
                assert False, "Unknown symbol!"

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(
            dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
        )

        return dataset
