import gradio as gr
from gradio.components import Component
from manager import Manager
from typing import Dict, Any
from utils import gen_config, read_yaml_file, flatten_dict, TeeStream
from components.constants import CONPONENTS2ARGKEY, METHOD2PIPELINE
import time
import os
import yaml
from datetime import datetime
import sys
import json
import threading
from queue import Queue
import time


class Runner:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager
        self.pipeline_config = None
        self.pipeline = None
        self.component2argkey = CONPONENTS2ARGKEY
        self.method2pipeline = METHOD2PIPELINE

    def _parse_pipeline_args(self, data: Dict["Component", Any]) -> Dict[str, Any]:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]

        basic_setting = dict(
            method_name=get("basic.method_name"),
            gpu_id=get("basic.gpu_id"),
            framework=get("basic.framework"),
            generator_model=get("basic.generator_name"),
            generator_model_path=get("basic.generator_model_path"),
            retrieval_method=get("basic.retrieval_method"),
            retrieval_model_path=get("basic.retrieval_model_path"),
            corpus_path=get("basic.corpus_path"),
            index_path=get("basic.index_path"),
        )

        retriever_settring = dict(
            # Retriever Setting
            instruction=get("retrieve.instruction"),
            retrieval_topk=int(get("retrieve.retrieval_topk")),
            retrieval_batch_size=int(get("retrieve.retrieval_batch_size")),
            retrieval_use_fp16=bool(get("retrieve.retrieval_use_fp16")),
            retrieval_query_max_length=int(get("retrieve.query_max_length")),
            save_retrieval_cache=get("retrieve.save_retrieval_cache"),
            use_retrieval_cache=get("retrieve.use_retrieval_cache"),
            retrieval_cache_path=get("retrieve.retrieval_cache_path"),
            retrieval_pooling_method=get("retrieve.retrieval_pooling_method"),
            bm25_backend=get("retrieve.bm25_backend").lower(),
            use_sentence_transformer=get("retrieve.use_sentence_transformers"),
        )

        reranker_setting = dict(
            # Reranker Setting
            use_reranker=bool(get("rerank.use_rerank")),
            rerank_model_name=get("rerank.rerank_model_name"),
            rerank_model_path=get("rerank.rerank_model_path"),
            rerank_pooling_method=get("rerank.rerank_pooling_method"),
            rerank_topk=int(get("rerank.rerank_topk")),
            rerank_max_length=int(get("rerank.rerank_max_len")),
            rerank_use_fp16=get("rerank.rerank_use_fp16"),
        )

        generator_setting = dict(
            # Generator Setting
            generator_max_input_len=int(get("generate.generator_max_input_len")),
            generator_batch_size=int(get("generate.generator_batch_size")),
            gpu_memory_utilization=float(get("generate.gpu_memory_utilization")),
            use_fid=get("generate.generate_use_fid"),
        )

        openai_setting = dict(
            api_key=get("generate.api_key"),
            base_url=get("generate.base_url"),
        )

        generation_params = dict(
            do_sample=get("generate.generate_do_sample"),
            max_tokens=int(get("generate.generate_max_new_tokens")),
            temperature=float(get("generate.generate_temperature")),
            top_p=float(get("generate.generate_top_p")),
            top_k=int(get("generate.generate_top_k")),
        )

        llmlingua_setting = dict(
            refiner_name="longllmlingua",
            refiner_model_path=get("method.llmlingua_refiner_path"),
            llmlingua_config=dict(
                refiner_input_prompt_flag=get("method.llmlingua_refiner_input_prompt_flag"),
                rate=get("method.llmlingua_rate"),
                llmlingua_target_token=get("method.llmlingua_target_token"),
                condition_in_question=get("method.llmlingua_condition_in_question"),
                reorder_context=get("method.llmlingua_reorder_context"),
                condition_compare=get("method.llmlingua_condition_compare"),
                context_budget=get("method.llmlingua_context_budget"),
                rank_method=get("method.llmlingua_rank_method"),
                force_tokens=get("method.llmlingua_force_tokens"),
                chunk_end_tokens=get("method.llmlingua_chunk_end_tokens"),
            ),
        )

        recomp_setting = dict(
            refiner_name="recomp",
            refiner_model_path=get("method.recomp_refiner_path"),
            refiner_max_input_length=get("method.recomp_max_input_length"),
            refiner_max_output_length=get("method.recomp_max_output_length"),
            refiner_topk=get("method.recomp_topk"),
            refiner_pooling_method=get("method.recomp_refiner_pooling_method"),
            refiner_encode_max_length=get("method.recomp_encode_max_length"),
        )

        sc_setting = dict(
            refiner_name="selective-context",
            refiner_model_path=get("method.sc_refiner_path"),
            sc_config={
                "reduce_ratio": float(get("method.sc_reduce_ratio")),
                "reduce_level": get("method.sc_reduce_level"),
            },
        )

        retrobust_setting = dict(
            generator_lora_path=get("method.retrobust_generator_lora_path"),
            max_iter=int(get("method.retrobust_max_iter")),
            single_hop=bool(get("method.retrobust_single_hop")),
        )

        skr_setting = dict(
            judger_name="skr",
            judger_config={
                "model_path": get("method.skr_judger_path"),
                "training_data_path": get("method.skr_training_data_path"),
                "topk": get("method.skr_topk"),
                "batch_size": get("method.skr_batch_size"),
                "max_length": get("method.skr_max_length"),
            },
        )

        selfrag_setting = dict(
            self_rag_setting={
                "mode": get("method.selfrag_mode"),
                "threshold": get("method.selfrag_threshold"),
                "max_depth": get("method.selfrag_max_depth"),
                "beam_width": get("method.selfrag_beam_width"),
                "w_rel": get("method.selfrag_w_rel"),
                "w_sup": get("method.selfrag_w_sup"),
                "w_use": get("method.selfrag_w_use"),
                "use_grounding": get("method.selfrag_use_grounding"),
                "use_utility": get("method.selfrag_use_utility"),
                "use_seqscore": get("method.selfrag_use_seqscore"),
                "ignore_cont": get("method.selfrag_ignore_cont"),
            },
            generation_params={
                "max_tokens": 100,
                "temperature": 0.0,
                "top_p": 1.0,
                "skip_special_tokens": False,
            },
        )

        flare_setting = dict(
            threshold=get("method.flare_threshold"),
            look_ahead_steps=get("method.flare_look_ahead_steps"),
            max_generation_length=get("method.flare_max_generation_length"),
            max_iter_num=get("method.flare_max_iter_num"),
        )

        iterretgen_setting = dict(iter_num=get("method.iterretgen_iter_num"))

        ircot_setting = dict(max_iter=get("method.ircot_max_iter"))

        trace_setting = dict(
            num_examplars=int(get("method.trace_num_examplars")),
            max_chain_length=int(get("method.trace_max_chain_length")),
            topk_triple_select=get("method.trace_topk_triple_select"),
            num_choices=get("method.trace_num_choices"),
            min_triple_prob=get("method.trace_min_triple_prob"),
            num_beams=get("method.trace_num_beams"),
            num_chains=get("method.trace_num_choices"),
            n_context=get("method.trace_n_context"),
            context_type=get("method.trace_context_type"),
        )

        spring_setting = dict(token_embedding_path=get("method.spring_token_embedding_path"))

        adaptive_setting = dict(
            judger_name="adaptive",
            judger_config=dict(model_path=get("method.adaptive_judger_path")),
        )

        rqrag_setting = dict(max_depth=get("method.rqrag_max_depth"))

        args = dict()
        args.update(basic_setting)
        args.update(retriever_settring)
        args.update(reranker_setting)
        args.update(generator_setting)
        openai_setting = {k: v for k, v in openai_setting.items() if v != ""}
        args["openai_setting"] = openai_setting
        args["generation_params"] = generation_params

        # set default setting
        for key in [
            "index_path",
            "corpus_path",
            "retrieval_model_path",
            "retrieval_pooling_method",
            "instruction",
            "retrieval_cache_path",
            "rerank_model_name",
            "rerank_model_path",
        ]:
            if key in args and args[key] == "":
                args[key] = None

        method2setting = {
            "LongLLMLingua": llmlingua_setting,
            "Recomp": recomp_setting,
            "Selective-Context": sc_setting,
            "Ret-Robust": retrobust_setting,
            "Skr": skr_setting,
            "Self-RAG": selfrag_setting,
            "Flare": flare_setting,
            "Iter-Retgen": iterretgen_setting,
            "IRCOT": ircot_setting,
            "Trace": trace_setting,
            "Spring": spring_setting,
            "Adaptive-RAG": adaptive_setting,
            "RQ-RAG": rqrag_setting,
        }

        if args["method_name"] in method2setting:
            args.update(method2setting[args["method_name"]])
            args[args["method_name"]] = method2setting[args["method_name"]]

        return args

    def _parse_evaluate_args(self, data: Dict["Component", Any]):
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        args = dict()

        # evaluate configs
        dataset_name = get("evaluate.dataset_name")
        data_dir = get("evaluate.data_dir")
        evaluate_setting = dict(
            dataset_name=dataset_name,
            data_dir=data_dir,
            dataset_split=get("evaluate.dataset_split"),
            save_intermediate_data=get("evaluate.save_intermediate_data"),
            save_dir=get("evaluate.save_dir"),
            save_note=get("evaluate.save_note"),
            seed=get("evaluate.seed"),
            test_sample_num=get("evaluate.test_sample_num"),
            random_sample=get("evaluate.random_sample"),
            metrics=get("evaluate.use_metrics"),
            save_metric_score=get("evaluate.save_metric_score"),
        )
        args.update(evaluate_setting)

        return args

    def _preivew_pipeline(self, data: Dict["Component", Any]):
        print("Previewing...")
        output_box = self.manager.get_elem_by_id("{}.output_box".format("preview"))
        args = self._parse_pipeline_args(data)
        yield {output_box: gen_config(args)}

    def preview_pipeline_configs(self, data):
        yield from self._preivew_pipeline(data)

    def _preivew_eval(self, data: Dict["Component", Any]):
        print("Previewing...")
        output_box = self.manager.get_elem_by_id("{}.evaluate_output_box".format("evaluate"))
        args = self._parse_evaluate_args(data)
        yield {output_box: gen_config(args)}

    def preview_eval_configs(self, data):
        yield from self._preivew_eval(data)

    def get_data_subfolders(self, folder_path: str):
        """Used in evaluate module to update the dataset_name dropdown."""

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
            if subfolders:
                return gr.update(choices=subfolders, value=subfolders[0])
            else:
                return gr.update(choices=[], value=None)
        else:
            return gr.update(choices=[], value=None)

    def get_dataset_split(self, data_dir: str, dataset_name: str):
        """Get the dataset split"""
        if not (os.path.exists(data_dir) and os.path.isdir(data_dir)):
            return gr.update(choices=[], value=None)

        if dataset_name is None:
            return gr.update(choices=[], value=None)

        dataset_dir = os.path.join(data_dir, dataset_name)

        if not (os.path.exists(dataset_dir) and os.path.isdir(dataset_dir)):
            return gr.update(choices=[], value=None)

        subfiles = [f.name for f in os.scandir(dataset_dir) if f.is_file()]
        valid_splits = ["train", "dev", "test"]
        final_splits = [split for split in valid_splits if any(f.startswith(split) for f in subfiles)]

        if final_splits:
            return gr.update(choices=final_splits, value=final_splits[0])

        return gr.update(choices=[], value=None)

    def get_config_files(self):
        """Used in basic module to get the saved config files."""
        current_path = os.getcwd()
        target_folder = "webui_configs"
        folder_path = os.path.join(current_path, target_folder)

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            file_list = os.listdir(folder_path)
            sorted_file_list = sorted(
                file_list,
                key=lambda x: datetime.strptime(
                    x.split("_")[1] + "_" + x.split("_")[2].split(".")[0],
                    "%Y-%m-%d_%H-%M-%S",
                ),
                reverse=True,
            )
            return sorted_file_list
        else:
            return []

    def save_configs(self, data):
        """Used in preview modulde to save the configs and write them to a file."""
        args = self._parse_pipeline_args(data)
        args.update(self._parse_evaluate_args(data))

        save_dir = os.path.join(os.getcwd(), "webui_configs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")
        time_str = current_time.strftime("%H-%M-%S")
        config_file_path = os.path.join(save_dir, f"config_{date_str}_{time_str}.yaml")
        with open(config_file_path, "w") as f:
            yaml.dump(args, f, default_flow_style=False, indent=4, sort_keys=False)
        save_message = f"Successfully saved configs to {config_file_path}"

        output_box = self.manager.get_elem_by_id("{}.output_box".format("preview"))
        yield {output_box: save_message}

    def update_config_file_list(self):
        """Used in preview module to update the config_file dropdown."""
        files = self.get_config_files()

        return gr.update(choices=files, value=files[0])

    def load_config_from_file(self, config_file_name: str):
        current_path = os.getcwd()
        file_path = os.path.join(current_path, "webui_configs", config_file_name)
        data = read_yaml_file(file_path)
        data = flatten_dict(data)

        new_config = {
            elem: elem.__class__(**{"value": data.get(self.component2argkey[elem_name], elem.value)})
            for elem_name, elem in self.manager.get_elem_iter()
            if elem_name in self.component2argkey
        }

        return new_config

    def _load_generator(self, config):
        from flashrag.utils import get_generator

        if config["method_name"].lower() != "replug":
            return get_generator(config)
        else:
            from flashrag.pipeline.replug_utils import load_replug_model

            model = load_replug_model(config["generator_model_path"])
            return get_generator(config, model=model)

    def _load_retriever(self, config):
        from flashrag.utils import get_retriever

        if config["method_name"] != "Vanila Generation":
            return get_retriever(config)
        else:
            return None

    def _load_pipeline_instance(self, config, generator, retriever):
        import importlib

        pipeline_name = self.method2pipeline[config["method_name"]]
        pipeline_cls = getattr(importlib.import_module("chat_pipelines"), pipeline_name)
        if config["method_name"] != "Vanila Generation":
            return pipeline_cls(config=config, generator=generator, retriever=retriever)
        else:
            from flashrag.prompt import PromptTemplate

            prompt_template = PromptTemplate(
                config=config,
                system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
                user_prompt="Question: {question}",
            )
            return pipeline_cls(
                config=config,
                prompt_template=prompt_template,
                generator=generator,
                retriever=retriever,
            )

    def _prepare_pipeline(self, config, progress=gr.Progress()):

        import torch

        if self.pipeline is None:
            self.pipeline_config = config

            progress(0.25, desc="Loading generator...")
            self.generator = self._load_generator(config)

            progress(0.5, desc="Loading retriever...")
            self.retriever = self._load_retriever(config)

            progress(0.75, desc="Loading pipeline...")
            self.pipeline = self._load_pipeline_instance(config, self.generator, self.retriever)

            progress(1, desc="Finished loading...")
        else:
            if config["generator_model_path"] != self.pipeline_config["generator_model_path"]:
                progress(0.25, desc="Reloading generator...")
                del self.generator
                torch.cuda.empty_cache()
                self.generator = self._load_generator(config)
            else:
                self.generator.config = config

            retriever_keys = ["retriever_model_path", "index_path", "corpus_path"]
            if any([config[key] != self.pipeline_config[key] for key in retriever_keys]) or (
                self.pipeline_config["method_name"] == "Vanila Generation"
                and self.pipeline_config["method_name"] != config["method_name"]
            ):
                progress(0.5, desc="Reloading retriever...")
                del self.retriever
                torch.cuda.empty_cache()
                self.retriever = self._load_retriever(config)
            else:
                if self.retriever is not None:
                    self.retriever.config = config
                else:
                    print("Retriever is None!")

            progress(0.75, desc="Reloading pipeline...")
            if config["method_name"] != self.pipeline_config["method_name"]:
                del self.pipeline
                torch.cuda.empty_cache()
                self.pipeline = self._load_pipeline_instance(config, self.generator, self.retriever)
                self.pipeline_config = config
            else:
                self.pipeline.config = config
                self.pipeline_config = config

            progress(1, desc="Finished reloading...")

    def load_pipeline(self, data: Dict["Component", Any], progress=gr.Progress()):
        from flashrag.config import Config

        progress(0, desc="Loading config...")

        # Load config
        args = self._parse_pipeline_args(data)
        args["disable_save"] = True
        config = Config(config_dict=args)

        # Load pipeline
        self._prepare_pipeline(config, progress)

        return self.manager.get_elem_by_id("chat.chatbot")

    def run_evaluate(self, data: Dict["Component", Any], progress=gr.Progress()):
        from flashrag.config import Config
        from flashrag.utils import get_dataset
        from flashrag.evaluator import Evaluator

        DONE_MARKER = "__DONE__"

        def run_task(queue):
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = TeeStream(queue, original_stdout)
            sys.stderr = TeeStream(queue, original_stderr)

            # Load config
            args = self._parse_pipeline_args(data)
            args.update(self._parse_evaluate_args(data))
            args["split"] = [args["dataset_split"]]
            args["disable_save"] = True
            config = Config(config_dict=args)
            print("Finish loading config...")

            print("# Args")
            print(json.dumps(args, indent=4, ensure_ascii=False))

            # Load data
            dataset = get_dataset(config)[args["dataset_split"]]
            print("Dataset info:")
            print(dataset)

            # Load pipeline
            self._prepare_pipeline(config, progress)
            self.pipeline.evaluator = Evaluator(config)

            # Run evaluation process
            result = self.pipeline.run(dataset)

            sys.stdout = original_stdout
            sys.stderr = original_stderr
            queue.put(DONE_MARKER)

        message_queue = Queue()
        output_history = "```bash"
        threading.Thread(target=run_task, args=(message_queue,), daemon=True).start()
        while True:
            while not message_queue.empty():
                message = message_queue.get()
                if message == DONE_MARKER:
                    yield output_history + "```"
                    return
                # 累积消息到历史记录
                output_history += message + "\n"
                yield output_history + "```"
            time.sleep(0.1)
