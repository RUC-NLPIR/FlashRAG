METHODS = ['Naive RAG', 'Vanila Generation','AAR', 'LongLLMLingua', 'Recomp', 'Selective-Context', 'Ret-Robust', 'Sure', 'RePlug', 'Skr', 'Self-RAG', 'Flare', 'Iter-Retgen', 'IRCOT', 'Trace', "Spring", 'Adaptive-RAG', 'RQ-RAG'
]

METRICS = [
    'em', 'f1' ,'acc', 'precision', 'recall',
    'input_tokens', 'bleu', 'rouge-l', 'rouge-1', 'rouge-2', 'zh_rouge-1', 'zh_rouge-2', 'zh_rouge-l'
]

CONPONENTS2ARGKEY = {
    # -------------- Basic Setting --------------
    "method_name": "method_name",
    "gpu_id": "gpu_id",
    "framework": "framework",
    "generator_name": "generator_model",
    "generator_model_path": "generator_model_path",
    "retrieval_method": "retrieval_method",
    "retrieval_model_path": "retrieval_model_path",
    "corpus_path": "corpus_path",
    "index_path": "index_path",

    # -------------- Retriever Setting --------------
    "instruction": "instruction",
    "retrieval_topk": "retrieval_topk",
    "retrieval_batch_size": "retrieval_batch_size",
    "retrieval_use_fp16": "retrieval_use_fp16",
    "query_max_length": "retrieval_query_max_length",
    "save_retrieval_cache": "save_retrieval_cache",
    "use_retrieval_cache": "use_retrieval_cache",
    "retrieval_cache_path": "retrieval_cache_path",
    "retrieval_pooling_method": "retrieval_pooling_method",
    "bm25_backend": "bm25_backend",
    "use_sentence_transformers": "use_sentence_transformer",

    # -------------- Reranker Setting --------------
    "use_rerank": "use_reranker",
    "rerank_model_name": "rerank_model_name",
    "rerank_model_path": "rerank_model_path",
    "rerank_pooling_method": "rerank_pooling_method",
    "rerank_topk": "rerank_topk",
    "rerank_max_len": "rerank_max_length",
    "rerank_use_fp16": "rerank_use_fp16",

    # -------------- Generator Setting --------------
    "generator_max_input_len": "generator_max_input_len",
    "generator_batch_size": "generator_batch_size",
    "gpu_memory_utilization": "gpu_memory_utilization",
    "generate_use_fid": "use_fid",

    # -------------- OpenAI Setting --------------
    "api_key": "openai_setting.api_key",
    "base_url": "opeani_setting.base_url",

    # -------------- generation_params --------------
    "generate_do_sample": "generation_params.do_sample",
    "generate_max_new_tokens": "generation_params.max_tokens",
    "generate_temperature": "generation_params.temperature",
    "generate_top_p": "generation_params.top_p",
    "generate_top_k": "generation_params.top_k",
    
    # -------------- Evaluatation Setting --------------
    "data_dir": "data_dir",
    "save_intermediate_data": "save_intermediate_data",
    "save_dir": "save_dir",
    "save_note": "save_note",
    "seed": "seed",
    "test_sample_num": "test_sample_num",
    "random_sample": "random_sample",
    "use_metrics": "metrics",
    "save_metric_score": "save_metric_score",
    
    
    
    
    # -------------- LLMlingua Setting --------------
    "llmlingua_refiner_path": "LongLLMLingua.refiner_model_path",
    "llmlingua_refiner_input_prompt_flag": "LongLLMLingua.llmlingua_config.refiner_input_prompt_flag",
    "llmlingua_rate": "LongLLMLingua.llmlingua_config.rate",
    "llmlingua_target_token": "LongLLMLingua.llmlingua_config.llmlingua_target_token",
    "llmlingua_condition_in_question": "LongLLMLingua.llmlingua_config.condition_in_question",
    "llmlingua_reorder_context": "LongLLMLingua.llmlingua_config.reorder_context",
    "llmlingua_condition_compare": "LongLLMLingua.llmlingua_config.condition_compare",
    "llmlingua_context_budget": "LongLLMLingua.llmlingua_config.context_budget",
    "llmlingua_rank_method": "LongLLMLingua.llmlingua_config.rank_method",
    "llmlingua_force_tokens": "LongLLMLingua.llmlingua_config.force_tokens",
    "llmlingua_chunk_end_tokens": "LongLLMLingua.llmlingua_config.chunk_end_tokens",
    
    # -------------- Recomp Setting --------------
    "recomp_refiner_path": "Recomp.refiner_model_path",
    "recomp_max_input_length": "Recomp.refiner_max_input_length",
    "recomp_max_output_length": "Recomp.refiner_max_output_length",
    "recomp_topk": "Recomp.refiner_topk",
    "recomp_refiner_pooling_method": "Recomp.refiner_pooling_method",
    "recomp_encode_max_length": "Recomp.refiner_encode_max_length",
    
    # -------------- Selective-Context Setting --------------
    "sc_refiner_path": "Selective-Context.refiner_model_path",
    "sc_reduce_ratio": "Selective-Context.sc_config.reduce_ratio",
    "sc_reduce_level": "Selective-Context.sc_config.reduce_level",
    
    # -------------- Retrobust Setting --------------
    "retrobust_generator_lora_path": "Ret-Robust.generator_lora_path",
    "retrobust_max_iter": "Ret-Robust.max_iter",
    "retrobust_single_hop": "Ret-Robust.single_hop",
    
    # -------------- SKR Setting --------------
    "skr_judger_path": "Skr.judger_config.model_path",
    "skr_training_data_path": "Skr.judger_config.training_data_path",
    "skr_topk": "Skr.judger_config.topk",
    "skr_batch_size": "Skr.judger_config.batch_size",
    "skr_max_length": "Skr.judger_config.max_length",
    
    # -------------- Self-RAG Setting --------------
    "selfrag_mode": "Self-RAG.self_rag_setting.mode",
    "selfrag_threshold": "Self-RAG.self_rag_setting.threshold",
    "selfrag_max_depth": "Self-RAG.self_rag_setting.max_depth",
    "selfrag_beam_width": "Self-RAG.self_rag_setting.beam_width",
    "selfrag_w_rel": "Self-RAG.self_rag_setting.w_rel",
    "selfrag_w_sup": "Self-RAG.self_rag_setting.w_sup",
    "selfrag_w_use": "Self-RAG.self_rag_setting.w_use",
    "selfrag_use_grounding": "Self-RAG.self_rag_setting.use_grounding",
    "selfrag_use_utility": "Self-RAG.self_rag_setting.use_utility",
    "selfrag_use_seqscore": "Self-RAG.self_rag_setting.use_seqscore",
    "selfrag_ignore_cont": "Self-RAG.self_rag_setting.ignore_cont",
    
    # -------------- FLARE Setting --------------
    "flare_threshold": "Flare.threshold",
    "flare_look_ahead_steps": "Flare.look_ahead_steps",
    "flare_max_generation_length": "Flare.max_generation_length",
    "flare_max_iter_num": "Flare.max_iter_num",
    
    # -------------- IterRetGen Setting --------------
    "iterretgen_iter_num": "Iter-Retgen.iter_num",
    
    # -------------- IRCOT Setting --------------
    "ircot_max_iter": "IRCOT.max_iter",
    
    # -------------- Trace Setting --------------
    "trace_num_examplars": "Trace.num_examplars",
    "trace_max_chain_length": "Trace.max_chain_length",
    "trace_topk_triple_select": "Trace.topk_triple_select",
    "trace_num_choices": "Trace.num_choices",
    "trace_min_triple_prob": "Trace.min_triple_prob",
    "trace_num_beams": "Trace.num_beams",
    "trace_n_context": "Trace.n_context",
    "trace_context_type": "Trace.context_type",
    
    # -------------- Spring Setting --------------
    "spring_token_embedding_path": "Spring.token_embedding_path",
    
    # -------------- Adaptive-RAG Setting --------------
    "adaptive_judger_path": "Adaptive-RAG.judger_config.model_path",
    
    # -------------- RQ-RAG Setting --------------
    "rqrag_max_depth": "RQ-RAG.max_depth"
}

METHOD2PIPELINE = {
    'Naive RAG': 'SequentialPipeline_Chat',
    'Vanila Generation': 'NaivePipeline_Chat',
    'AAR': 'SequentialPipeline_Chat',
    'LongLLMLingua': 'SequentialPipeline_Chat',
    'Recomp': 'SequentialPipeline_Chat',
    'Selective-Context': 'SequentialPipeline_Chat',
    'Ret-Robust': 'SelfAskPipeline_Chat',
    'Sure': 'SuRePipeline_Chat',
    'RePlug': 'REPLUGPipeline_Chat',
    'Skr': 'ConditionalPipeline_Chat',
    'Self-RAG': 'SelfRAGPipeline_Chat',
    'Flare': 'FLAREPipeline_Chat',
    'Iter-Retgen': 'IterativePipeline_Chat',
    'IRCOT': 'IRCOTPipeline_Chat',
    'Trace': 'SequentialPipeline_Chat',
    'Spring': 'SequentialPipeline_Chat',
    'Adaptive-RAG': 'AdaptivePipeline_Chat',
    'RQ-RAG': 'RQRAGPipeline_Chat'
}