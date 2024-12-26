import gradio as gr
from gradio.components import Component
from manager import Manager
from typing import Dict, Any, Generator
from utils import gen_config

class Runner:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager
    def _parse_args(self, data: Dict["Component", Any]) -> Dict[str, Any]:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        
        basic_setting = dict(
            method_name = get("basic.method_name"),
            gpu_id = get("basic.gpu_id"),
            framework = get("basic.framework"),
            generator_model = get("basic.generator_name"),
            generator = get("basic.generator_model_path"),
            retrieval_method = get("basic.retrieval_method"),
            retrieval_model_path = get("basic.retrieval_model_path"),
            corpus_path = get("basic.corpus_path"),
            index_path = get("basic.index_path"),
        )
        
        retriever_settring = dict(
            # Retriever Setting
            instruction = get("retrieve.instruction"),
            retrieval_topk = int(get("retrieve.retrieval_topk")),
            retrieval_batch_size = int(get("retrieve.retrieval_batch_size")),
            retrieval_use_fp16 = get("retrieve.retrieval_use_fp16"),
            retrieval_query_max_length = int(get("retrieve.query_max_length")),
            save_retrieval_cache = get("retrieve.save_retrieval_cache"),
            use_retrieval_cache = get("retrieve.use_retrieval_cache"),
            retrieval_cache_path = get("retrieve.retrieval_cache_path"),
            retrieval_pooling_method = get("retrieve.retrieval_pooling_method"),
            bm25_backend = get("retrieve.bm25_backend"),
            use_sentence_transformer = get("retrieve.use_sentence_transformers"),
        )
        
        reranker_setting = dict(
            # Reranker Setting
            use_reranker = get("rerank.use_rerank"),
            rerank_model_name = get("rerank.rerank_model_name"),
            rerank_model_path = get("rerank.rerank_model_path"),
            rerank_pooling_method = get("rerank.rerank_pooling_method"),
            rerank_topk = int(get("rerank.rerank_topk")),
            rerank_max_length = int(get("rerank.rerank_max_len")),
            rerank_use_fp16 = get("rerank.rerank_use_fp16"),
        )
        
        generator_setting = dict(
            # Generator Setting
            generator_max_input_len = int(get("generate.generator_max_input_len")),
            generator_batch_size = int(get("generate.generator_batch_size")),
            gpu_memory_utilization = float(get("generate.gpu_memory_utilization")),
            use_fid = get("generate.generate_use_fid"),
        )
        
        openai_setting = dict(
            api_key = get("generate.api_key"),
            base_url = get("generate.base_url"),
        )
        
        generation_params = dict(
            do_sample = get("generate.generate_do_sample"),
            max_tokens = int(get("generate.generate_max_new_tokens")),
            temperature = float(get("generate.generate_temperature")),
            top_p = float(get("generate.generate_top_p")),
            top_k = int(get("generate.generate_top_k"))
        )
        
        llmlingua_setting = dict(
            refiner_input_prompt_flag = get("method.llmlingua_refiner_input_prompt_flag"),
            rate = get("method.llmlingua_rate"),
            llmlingua_target_token = get("method.llmlingua_target_token"),
            condition_in_question = get("method.llmlingua_condition_in_question"),
            reorder_context = get("method.llmlingua_reorder_context"),
            condition_compare = get("method.llmlingua_condition_compare"),
            context_budget = get("method.llmlingua_context_budget"),
            rank_method = get("method.llmlingua_rank_method"),
            force_tokens = get("method.llmlingua_force_tokens"),
            chunk_end_tokens = get("method.llmlingua_chunk_end_tokens"),
            word_label = get("method.llmlingua_word_label"),
            drop_consecutive = get("method.llmlingua_drop_consecutive")
        )
        
        recomp_setting = dict(
            refiner_name = "recomp-abstractive",
            refiner_model_path = get("method.recomp_refiner_path"),
            refiner_max_input_length = get("method.recomp_max_input_length"),
            refiner_max_output_length = get("method.recomp_max_output_length"),
            refiner_topk = get("method.recomp_topk"),
            refiner_pooling_method = get("method.recomp_refiner_pooling_method"),
            refiner_encode_max_length = get("method.recomp_encode_max_length"),
        )
        
        sc_setting = dict(
            sc_reduce_ratio = float(get("method.sc_reduce_ratio")),
            sc_reduce_level = get("method.sc_reduce_level"),
        )
        
        retrobust_setting = dict(
            generator_lora_path = get("method.retrobust_generator_lora_path"),
            max_iter = int(get("method.retrobust_max_iter")),
            single_hop = get("method.retrobust_single_hop")
        )
        
        skr_setting = dict(
            model_path = get("method.skr_judger_path"),
            training_data_path = get("method.skr_training_data_path"),
            topk = get("method.skr_topk"),
            batch_size = get("method.skr_batch_size"),
            max_length = get("method.skr_max_length")
        )
        
        selfrag_setting = dict(
            mode = get("method.selfrag_mode"),
            threshold = get("method.selfrag_threshold"),
            max_depth = get("method.selfrag_max_depth"),
            beam_width = get("method.selfrag_beam_width"),
            w_rel = get("method.selfrag_w_rel"),
            w_sup = get("method.selfrag_w_sup"),
            w_use = get("method.selfrag_w_use"),
            use_grounding = get("method.selfrag_use_grounding"),
            use_utility = get("method.selfrag_use_utility"),
            use_seqscore = get("method.selfrag_use_seqscore"),
            ignore_cont = get("method.selfrag_ignore_cont")
        )       
        
        flare_setting = dict(
            threshold = get("method.flare_threshold"),
            look_ahead_steps = get("method.flare_look_ahead_steps"),
            max_generation_length = get("method.flare_max_generation_length"),
            max_iter_num = get("method.flare_max_iter_num"),
        )
        
        iterretgen_setting = dict(
            iter_num = get("method.iterretgen_iter_num")
        )
        
        ircot_setting = dict(
            max_iter = get("method.ircot_max_iter")
        )
        
        trace_setting = dict(
            num_examplars = int(get("method.trace_num_examplars")),
            max_chain_length = int(get("method.trace_max_chain_length")),
            topk_triple_select = get("method.trace_topk_triple_select"),
            num_choices = get("method.trace_num_choices"),
            min_triple_prob = get("method.trace_min_triple_prob"),
            num_beams = get("method.trace_num_beams"),
            num_chains = get("method.trace_num_choices"),
            n_context = get("method.trace_n_context"),
            context_type = get("method.trace_context_type"),
        )
        
        spring_setting = dict(
            token_embedding_path = get("method.spring_token_embedding_path")
        )
        
        adaptive_setting = dict(
            model_path = get("method.adaptive_judger_path")
        )
        
        rqrag_setting = dict(
            max_depth = get("method.rqrag_max_depth")
        )
        
        args = dict()
        args.update(basic_setting)
        args.update(retriever_settring)
        args.update(reranker_setting)
        args.update(generator_setting)
        args["openai_setting"] = openai_setting
        args["generation_params"] = generation_params

        if args["method_name"] == "LongLLMLingua":
            args["refiner_name"] = "longllmlingua"
            args["refiner_model_path"] = get("method.llmlingua_refiner_path"),
            args["llmlingua_config"] = llmlingua_setting
            
        elif args["method_name"] == "Recomp":
            args.update(recomp_setting)
            
        elif args["method_name"] == "Selective-Context":
            args["refiner_name"] = "Selective-Context"
            args["refiner_model_path"] = get("recomp.recomp_model_path")
            args["sc_config"] = sc_setting
            
        elif args["method_name"] == "Ret-Robust":
            args.update(retrobust_setting)
            
        elif args["method_name"] == "Skr":
            args["judger_name"] = get("method.skr_judger_name")
            args["judger_config"] = skr_setting
            
        elif args["method_name"] == "Self-RAG":
            args["generation_params"].update(dict(skip_special_tokens = False))
            args["selfrag_config"] = selfrag_setting
        
        elif args["method_name"] == "FLare":
            args["flare_config"] = flare_setting
        
        elif args["method_name"] == "Iter-Retgen":
            args["iterretgen_config"] = iterretgen_setting
        
        elif args["method_name"] == "IRCOT":
            args["ircot_config"] = ircot_setting
        
        elif args["method_name"] == "Trace":
            args["trace_setting"] = trace_setting

        elif args["method_name"] == "Spring":
            args.update(spring_setting)
    
        elif args["method_name"] == "Adaptive-RAG":
            args["judger_name"] = get("method.adaptive_judger_name")
            args["judger_config"] = adaptive_setting
            
        elif args["method_name"] == "RQ-RAG":
            args["rqrag_config"] = rqrag_setting
        
        return args
        
    def _preivew(self, data: Dict["Component", Any]):
        output_box = self.manager.get_elem_by_id("{}.output_box".format("preview"))
        print("Previewing...")
        args = self._parse_args(data)
        
        yield {output_box: gen_config(args)}
    def preview_configs(self, data):
        yield from self._preivew(data)