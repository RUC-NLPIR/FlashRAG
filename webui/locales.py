LOCALES = {
    "lang": {
        "en": {
            "label": "Language",
        },
        "zh": {
            "label": "语言"
        }
    },
    "method_name": {
        "en": {
            "label": "Method"
        },
        "zh": {
            "label": "方法"
        }
    },
    "gpu_id": {
        "en": {
            "label": "GPU ID",
            "value": "0,1,2,3,4,5,6,7,8"
        },
        "zh": {
            "label": "GPU 序号",
            "value": "0,1,2,3,4,5,6,7,8"
        }
    },
    "framework": {
        "en": {
            "label": "Framework"
        },
        "zh": {
            "label": "推理框架"
        }
    },
    "generator_name": {
        "en": {
            "label": "Generator Name",
            "value": "llama3.1-8B-instruct"
        },
        "zh": {
            "label": "生成器",
            "value": "llama3.1-8B-instruct"
        }
    },
    "generator_model_path": {
        "en": {
            "label": "Generator Model Path",
            "value": "meta-llama/Llama-3.1-8B-Instruct"
        },
        "zh": {
            "label": "生成器模型路径",
            "value": "meta-llama/Llama-3.1-8B-Instruct"
        }
    },
    "retrieval_method": {
        "en": {
            "label": "Retrieval Method",
            "value": "e5"
        },
        "zh": {
            "label": "检索方法",
            "value": "e5"
        }
    },
    "retrieval_model_path": {
        "en": {
            "label": "Retrieval Model Path",
            "value": "intfloat/e5-base-v2"
        },
        "zh": {
            "label": "检索模型路径",
            "value": "intfloat/e5-base-v2"
        }
    },
    "corpus_path": {
        "en": {
            "label": "Corpus Path",
            "value": "examples/quick_start/general_knowledge.jsonl"
        },
        "zh": {
            "label": "语料库路径",
            "value": "examples/quick_start/general_knowledge.jsonl"
        },
    },
    "index_path": {
        "en": {
            "label": "Index Path",
            "value": "examples/quick_start/e5_Flat.index"
        },
        "zh": {
            "label": "索引库路径",
            "value": "examples/quick_start/e5_Flat.index"
        }
    },
    "rerank_tab": {
        "en": {
            "label": "Rerank Settings"
        },
        "zh": {
            "label":"重排器选项卡"
        }
    },
    "base_tab": {
        'en': {
            "label": "Basic Settings"
        },
        "zh": {
            "label": "基础选项卡"
        }
    },
    "use_rerank": {
        "en": {
            "label": "Use Rerank",
            "info": "Whether to use documents reranking method"
        },
        "zh": {
            "label": "使用重排器",
            "info": "是否使用文档重排器"
        }
    },
    "rerank_model_name": {
        "en": {
            "label": "Rerank Model Name",
            "value": "bge-reranker-v2-m3"
        },
        "zh": {
            "label":"重排器模型",
            "value": "bge-reranker-v2-m3"
        }
    },
    "rerank_model_path": {
        "en": {
            "label": "Rerank Model Path",
            "value": "BAAI/bge-reranker-v2-m3"
        },
        "zh": {
            "label":"重排器模型路径",
            "value": "BAAI/bge-reranker-v2-m3"
        }
    },
    "rerank_pooling_method": {
        "en": {
            "label": "Rerank Pooling Method",
            "info": "Only used in the case of using embedding model"
        },
        "zh": {
            "label":"重排器池化方法",
            "info":"仅在使用嵌入模型时使用"
        }
    },
    "rerank_topk": {
        "en": {
            "label": "Rerank TopK",
            "info": "The remain document number after rerank"
        },
        "zh": {
            "label":"重排TopK文档",
            "info":"重排后剩余文档数量"
        }
    },
    "rerank_max_len": {
        "en": {
            "label": "Reranker Max Input Length"
        },
        "zh": {
            "label":"重排器最大输入长度"
        }
    },
    "rerank_use_fp16": {
        "en": {
            "label": "fp16",
            "info": "Whether to use FP16 precision for reranker"
        },
        "zh": {
            "label":"fp16",
            "info": "重排器是否使用FP16精度"
        }
    },
    "instruction": {
        "en": {
            "label": "Instruction",
            "info": "Default instruction for retriever, only applicable to specific retrievers. For example, in the case of 'e5', use the prefix 'query:'.",
        },
        "zh": {
            "label": "检索器默认指令",
            "info": "检索器默认指令，仅适用于特定检索器。例如，对于'e5'，使用前缀'query:'",
        }
    },
    "retrieval_topk": {
        "en": {
            "label": "Top-K",
            "info": "The number of retrieved documents"
        },
        "zh": {
            "label": "Top-K",
            "info": "检索的文档数"
        }
    },
    "retrieval_use_fp16": {
        "en": {
            "label": "FP16",
            "info": "Whether to use FP16 precision for retrieval"
        },
        "zh": {
            "label": "FP16",
            "info": "是否使用FP16精度进行检索"
        }
    },
    "retrieval_pooling_method": {
        "en": {
            "label": "Pooling Method",
            "info": "The pooling method used for retrieval (e.g., mean, max, pooling)"
        },
        "zh": {
            "label": "池化方法",
            "info": "检索时使用的池化方法（如 mean, max, pooling）"
        }
    },
    "query_max_length": {
        "en": {
            "label": "Query Max Length",
            "info": "The maximum length of the query"
        },
        "zh": {
            "label": "查询最大长度",
            "info": "输入查询的最大长度, 过长的查询会被截断"
        }
    },
    "retrieval_batch_size": {
        "en": {
            "label": "Batch Size",
            "info": "The batch size used for retrieval."
        },
        "zh": {
            "label": "批量大小",
            "info": "检索时使用的批量大小。"
        }
    },
    "bm25_backend": {
        "en": {
            "label": "BM25 Backend",
            "info": "The backend used for BM25 retrieval."
        },
        "zh": {
            "label": "BM25后端",
            "info": "用于BM25检索的后端。"
        }
    },
    "use_sentence_transformers": {
        "en": {
            "label": "Use Sentence Transformers",
            "info": "Whether to use Sentence Transformers for retrieval."
        },
        "zh": {
            "label": "使用Sentence Transformers",
            "info": "是否使用 Sentence Transformers 作为检索后端"
        }
    },
    "save_retrieval_cache": {
        "en": {
            "label": "Save Retrieval Cache",
            "info": "Whether to save the retrieval cache"
        },
        "zh": {
            "label": "保存检索缓存",
            "info": "是否保存检索缓存"
        }
    },
    "use_retrieval_cache": {
        "en": {
            "label": "Use Retrieval Cache",
            "info": "Whether to use the retrieval cache"
        },
        "zh": {
            "label": "使用检索缓存",
            "info": "是否使用检索缓存"
        }
    },
    "retrieval_cache_path": {
        "en": {
            "label": "Retrieval Cache Path",
            "info": "The file path to save or load the retrieval cache"
        },
        "zh": {
            "label": "检索缓存路径",
            "info": "保存或加载检索缓存的文件路径"
        }
    },
    "retrieve_tab": {
        "en": {
            "label": "Retrieve Settings"
        },
        "zh": {
            "label":"检索器选项卡"
        }
    },
    "generate_tab": {
        "en": {
            "label": "Generate Settings"
        },
        "zh": {
            "label":"生成器选项卡"
        }
    },
    "openai_tab": {
        "en": {
            "label": "OpenAI Settings"
        },
        "zh": {
            "label": "OpenAI API 相关设置"
        }
    },
    "api_key": {
        "en": {
            "label": "API Key"
        },
        "zh": {
            "label":"API密钥"
        }
    },
    "base_url": {
        "en": {
            "label": "Base URL"
        },
        "zh": {
            "label": "Base URL"
        }
    },
    "generator_max_input_len": {
        "en": {
            "label": "Max Input Length"
        },
        "zh": {
            "label":"生成器最大输入长度"
        }
    },
    "generator_batch_size": {
        "en": {
            "label": "Batch Size"
        },
        "zh": {
            "label":"批量大小"
        }
    },
    "gpu_memory_utilization": {
        "en": {
            "label": "GPU Memory Utilization",
            "info": "Only valid for VLLM framework"
        },
        "zh": {
            "label":"GPU内存利用率",
            "info": "仅在VLLM框架下有效"
        }
    },
    "generate_do_sample": {
        "en": {
            "label": "Do Sample",
            "info": "Whether to sample from the output distribution."
        },
        "zh": {
            "label":"采样",
            "info": "是否从输出分布中采样。"
        }
    },
    "generate_max_new_tokens": {
        "en": {
            "label": "Max New Tokens"
        },
        "zh": {
            "label":"最大输出长度"
        }
    },
    "generate_use_fid": {
        "en": {
            "label": "Use FID",
            "info": "Whether to use FID, only valid in encoder-decoder model."
        },
        "zh": {
            "label":"使用FID",
            "info": "是否使用FID，仅在编码器-解码器模型有效。"
        }
    },
    "generate_temperature": {
        "en": {
            "label": "Temperature"
        },
        "zh": {
            "label":"温度"
        }
    },
    "generate_top_p": {
        "en": {
            "label": "Top P"
        },
        "zh": {
            "label":"Top P"
        }
    },
    "generate_top_k": {
        "en": {
            "label": "Top K"
        },
        "zh": {
            "label":"Top K"
        }
    },
    "config_preview_btn": {
        "en": {
            "value": "Config Preview"
        },
        "zh": {
            "value": "预览配置"
        }
    },
    "config_load_btn": {
        "en": {
            "value": "Load Config"
        },
        "zh": {
            "value": "加载配置"
        }
    },
    "config_save_btn": {
        "en": {
            "value": "Save Config"
        },
        "zh": {
            "value": "保存配置"
        }
    },
    "llmlingua_refiner_path": {
        "en": {
            "label": "LLMLingua Refiner Path",
            "info": "Path to the LLM lingua refiner model.",
            "value": "meta-llama/Llama-2-7b-hf"
        },
        "zh": {
            "label": "LLMLingua 精炼模型路径",
            "info": "LLM Lingua 精炼器模型的路径。",
            "value": "meta-llama/Llama-2-7b-hf"
        }
    },
    "llmlingua_use_llmlingua2": {
        "en": {
            "label": "Use LLMLingua 2",
            'info': 'LLMLingua2 is the next version of LLMLingua, which is faster. Refiner path for LLMLingua 2 should be set to trained bert model.'
        },
       "zh": {
           "label": "使用 LLMLingua 2",
           "info": "LLMLingua 2 是 LLMLingua 的下一版本，速度更快。LLMLingua 2 的精炼器路径应设置为训练好的 bert 模型。"
       }
    },
    "llmlingua_refiner_input_prompt_flag": {
        "en": {
            "label": "Input Prompt Flag",
            "info": "Whether to use input prompt as the input of refiner"
        },
        "zh": {
            "label": "输入提示标志",
            "info": "是否使用提示作为精炼器的输入"
        }
    },
    "llmlingua_rate": {
        "en": {
            "label": "Compression Rate",
            "info": "The actual compression rate is generally lower than the specified target, but there can be fluctuations due to differences in tokenizers."
        },
        "zh": {
            "label": "压缩比率",
            "info": "默认为0.5。实际的压缩比率通常低于指定的目标，但由于不同tokenizer的差异，可能会有波动。"
        }
    },
    "llmlingua_target_token": {
        "en": {
            "label": "Target Token Num",
            "info": "The global maximum number of tokens to be achieved"
        },
        "zh": {
            "label": "目标Token数",
            "info": "目标保留的token数量"
        }
    },
    "llmlingua_condition_in_question": {
        "en": {
            "label": "Whether Condition in Question",
            "info": "Specific condition to apply to question in the context",
            "value": "none"
        },
        "zh": {
            "label": "是否使用question作为条件",
            "info": "在上下文中适用于问题的具体条件",
           "value": "none"
        }
    },
    "llmlingua_reorder_context": {
        "en": {
            "label": "Reorder Context",
            "value": "original",
            "info": "Strategy for reordering context in the compressed result"
        },
        "zh": {
            "label": "重排序上下文",
            "value": "original",
            "info": "重排序上下文的策略"
        }
    },
    "llmlingua_condition_compare": {
        "en": {
            "label": "Condition Compare",
            "info": "Whether to enable condition comparison during token-level compression",
            "value": False
        },
        "zh": {
            "label": "条件比较",
            "info": "是否在分词级别进行条件比较",
            "value": False
        }
    },
    "llmlingua_context_budget": {
        "en": {
            "label": "Context Budget",
            "info": "Token budget for the context-level filtering, expressed as a string to indicate flexibility",
            "value": "+100"
        },
        "zh": {
            "label": "上下文预算",
            "info": "上下文级别的过滤的Token预算，以字符串形式表示可变性",
            "value": "+100"
        }
    },
    "llmlingua_rank_method": {
        "en": {
            "label": "Rank Method",
            "info": "Method used for ranking elements during compression.",
        },
        "zh": {
            "label": "排序方法",
            "info": "LLMLingua操作中使用的排序方法。"
        }
    },
    "llmlingua_force_tokens": {
        "en": {
            "label": "Force Tokens",
            "info": "List of specific tokens to always include in the compressed result",
            "value": []
        },
        "zh": {
            "label": "强制Tokens",
            "info": "强制包含在压缩结果中的特定Token列表",
            "value": []
        }
    },
    "llmlingua_chunk_end_tokens": {
        "en": {
            "label": "Chunk End Tokens",
            "info": "The early stop tokens for segmenting chunk",
            "value": [".", "\n"]
        },
        "zh": {
            "label": "块结束Tokens",
            "info": "分块的结束标记",
            "value": ["。", "\n"]
        }
    },
    "recomp_refiner_path": {
        "en": {
            "label": "Refiner Path",
            "value": "fangyuan/nq_abstractive_compressor"
        },
        "zh": {
            "label": "精炼器路径",
            "value": "fangyuan/nq_abstractive_compressor"
        }
    },
    "recomp_max_input_length": {
        "en": {
            "label": "Max Input Length",
            "info": "Maximum input length for Recomp refiner"
        },
        "zh": {
            "label": "最大输入长度",
            "info": "精炼器的最大输入长度"
        }
    },
    "recomp_max_output_length": {
        "en": {
            "label": "Max Output Length",
            "info": "Maximum output length for Recomp refiner (only used in abstractive refiner)"
        },
        "zh": {
            "label": "最大输出长度",
            "info": "精炼器的最大输出长度 (仅用于摘要精炼器)"
        }
    },
    "recomp_topk": {
        "en": {
            "label": "Top K",
            "info": "Number of top-k results for Recomp refiner (only used in extractive refiner)"
        },
        "zh": {
            "label": "Top K",
            "info": "精炼器的Top-K结果数量 (仅用于抽取式精炼器)"
        }
    },
    "recomp_encode_max_length": {
        "en": {
            "label": "Max Encode Length",
            "info": "Maximum encode length for recomp refiner (only used in extractive refiner)"
        },
        "zh": {
            "label": "最大编码长度",
            "info": "精炼器的最大编码长度 (仅用于抽取式精炼器)"
        }
    },
    "recomp_refiner_pooling_method": {
        "en": {
            "label": "Pooling Method",
            "info": "Pooling method used in Recomp refiner (only used in extractive refiner)"
        },
        "zh": {
            "label": "池化方法",
            "info": "精炼器使用的池化方法 (仅用于抽取式精炼器)"
        }
    },
    "sc_refiner_path": {
        "en": {
            "label": "Refiner Path",
            "value": "openai-community/gpt2"
        },
        "zh": {
            "label": "精炼器模型路径",
            "value": "openai-community/gpt2"
        }
    },
    "sc_reduce_ratio": {
        "en": {
            "label": "Reduce Ratio",
        },
        "zh": {
            "label": "压缩比例",
        }
    },
    "sc_reduce_level": {
        "en": {
            "label": "Reduce Level",
            "info": "Reduction level for SC refiner"
        },
        "zh": {
            "label": "压缩等级",
            "info": "Selective-Context精炼器的压缩粒度"
        }
    },
    "retrobust_generator_lora_path": {
        "en": {
            "label": "Generator LoRA Path",
            "info": "Path to the Retrobust generator LoRA model.",
            "value": "Ori/llama-2-13b-peft-nq-retrobust"
        },
        "zh": {
            "label": "生成器LoRA路径",
            "info": "Retrobust生成器使用的LoRA模型的路径。",
            "value": "Ori/llama-2-13b-peft-nq-retrobust"
        }
    },
    "retrobust_max_iter": {
        "en": {
            "label": "Max Iterations",
        },
        "zh": {
            "label": "最大迭代次数",
        }
    },
    "retrobust_single_hop": {
        "en": {
            "label": "Single Hop",
            "info": "Whether uses single-hop generation mode"
        },
        "zh": {
            "label": "单跳生成",
            "info": "是否使用单跳模式"
        }
    },
    "skr_judger_path": {
        "en": {
            "label": "Judger Path",
            "info": "Path to the SKR judger model"
        },
        "zh": {
            "label": "评判器路径",
            "info": "SKR评判器模型的路径"
        }
    },
    "skr_training_data_path": {
        "en": {
            "label": "Training Data Path",
            "info": "Path to the training data for SKR",
            "value": "examples/methods/sample_data/skr_training.json"
        },
        "zh": {
            "label": "训练数据路径",
            "info": "SKR的训练数据路径",
            "value": "examples/methods/sample_data/skr_training.json"
        }
    },
    "skr_topk": {
        "en": {
            "label": "Top K",
            "info": "Number of top-k results for SKR"
        },
        "zh": {
            "label": "Top K",
            "info": "SKR使用的Top-K结果数量"
        }
    },
    "skr_batch_size": {
        "en": {
            "label": "Batch Size",
            "info": "Batch size for SKR"
        },
        "zh": {
            "label": "批量大小",
            "info": "SKR的批处理大小"
        }
    },
    "skr_max_length": {
        "en": {
            "label": "Max Length",
            "info": "Maximum sequence length for SKR"
        },
        "zh": {
            "label": "最大长度",
            "info": "SKR的最大序列长度"
        }
    },
    "selfrag_mode": {
        "en": {
            "label": "Self-Rag Mode",
        },
        "zh": {
            "label": "Self-RAG模式",
        }
    },
    "selfrag_threshold": {
        "en": {
            "label": "Threshold",
            "info": "Retrieval Threshold used in Self-RAG"
        },
        "zh": {
            "label": "阈值",
            "info": "Self-RAG的检索阈值"
        }
    },
    "selfrag_max_depth": {
        "en": {
            "label": "Maximum Depth",
            "info": "The maximum depth of Self-Rag operations"
        },
        "zh": {
            "label": "最大深度",
            "info": "Self-Rag 最大迭代深度"
        }
    },
    "selfrag_beam_width": {
        "en": {
            "label": "Beam Width",
        },
        "zh": {
            "label": "Beam宽度",
        }
    },
    "selfrag_w_rel": {
        "en": {
            "label": "Relevance Weight",
        },
        "zh": {
            "label": "相关性权重",
        }
    },
    "selfrag_w_sup": {
        "en": {
            "label": "Support Weight",
        },
        "zh": {
            "label": "支持性权重",
        }
    },
    "selfrag_w_use": {
        "en": {
            "label": "Utility Weight",
        },
        "zh": {
            "label": "效用权重",
        }
    },
    "selfrag_use_grounding": {
        "en": {
            "label": "Use Grounding",
        },
        "zh": {
            "label": "使用 Grounding",
        }
    },
    "selfrag_use_utility": {
        "en": {
            "label": "Use Utility",
        },
        "zh": {
            "label": "使用效用评分",
        }
    },
    "selfrag_use_seqscore": {
        "en": {
            "label": "Use Sequence Score",
        },
        "zh": {
            "label": "使用序列评分",
        }
    },
    "selfrag_ignore_cont": {
        "en": {
            "label": "Ignore Continuity",
        },
        "zh": {
            "label": "忽略连续性",
        }
    },
    "flare_threshold": {
        "en": {
            "label": "Flare Threshold",
            "info": "The threshold for triggering flare operations"
        },
        "zh": {
            "label": "Flare阈值",
            "info": "触发flare检索操作的阈值"
        }
    },
    "flare_look_ahead_steps": {
        "en": {
            "label": "Look-Ahead Steps",
            "info": "Number of look-ahead steps for flare operations"
        },
        "zh": {
            "label": "预览步数",
            "info": "flare操作中的预览步数"
        }
    },
    "flare_max_generation_length": {
        "en": {
            "label": "Max Generation Length",
            "info": "Maximum length of generated sequences in flare operations"
        },
        "zh": {
            "label": "最大生成长度",
            "info": "flare中生成序列的最大长度"
        }
    },
    "flare_max_iter_num": {
        "en": {
            "label": "Max Iterations",
            "info": "Maximum number of iterations in flare operations"
        },
        "zh": {
            "label": "最大迭代次数",
            "info": "flare操作的最大迭代次数"
        }
    },
    "iterretgen_iter_num": {
        "en": {
            "label": "Iteration Number",
        },
        "zh": {
            "label": "迭代次数",
        }
    },
    "ircot_max_iter": {
        "en": {
            "label": "IRCOT Max Iterations",
            "info": "Maximum number of iterations for IRCOT operations"
        },
        "zh": {
            "label": "IRCOT最大迭代次数",
            "info": "IRCOT操作的最大迭代次数"
        }
    },
    "trace_num_examplars": {
        "en": {
            "label": "Number of Exemplars",
            "info": "Number of exemplars used in Trace"
        },
        "zh": {
            "label": "样本数量",
            "info": "trace中使用的few-shot样本数量"
        }
    },
    "trace_max_chain_length": {
        "en": {
            "label": "Max Chain Length",
        },
        "zh": {
            "label": "最大链长度",
        }
    },
    "trace_topk_triple_select": {
        "en": {
            "label": "Top-K Triple Selection",
        },
        "zh": {
            "label": "Top-K三元组选择",
        }
    },
    "trace_num_choices": {
        "en": {
            "label": "Number of Choices",
        },
        "zh": {
            "label": "选择数量",
        }
    },
    "trace_min_triple_prob": {
        "en": {
            "label": "Min Triple Probability",
        },
        "zh": {
            "label": "最小三元组概率",
        }
    },
    "trace_num_beams": {
        "en": {
            "label": "Number of Beams",
        },
        "zh": {
            "label": "Beam数量",
        }
    },
    "trace_num_chains": {
        "en": {
            "label": "Number of Chains",
        },
        "zh": {
            "label": "链数量",
        }
    },
    "trace_n_context": {
        "en": {
            "label": "Context Count",
        },
        "zh": {
            "label": "上下文数量",
        }
    },
    "trace_context_type": {
        "en": {
            "label": "Context Type",
        },
        "zh": {
            "label": "上下文类型",
        }
    },
    "spring_token_embedding_path": {
        "en": {
            "label": "Token Embedding Path",
            "info": "Path to the training token embedding for Spring"
        },
        "zh": {
            "label": "Token嵌入路径",
            "info": "Spring方法中训练好的Token嵌入路径"
        }
    },
    "adaptive_judger_path": {
        "en": {
            "label": "Judger Path",
        },
        "zh": {
            "label": "判断器路径",
        }
    },
    "rqrag_max_depth": {
        "en": {
            "label": "Max Depth",
            "info": "The maximum depth for rq-rag operations"
        },
        "zh": {
            "label": "最大深度",
            "info": "rq-rag 操作的最大深度"
        }
    },
    "submit_button": {
        "en": {
            "value": "Submit"
        },
        "zh": {
            "value": "提交"
        }
    },
    "data_dir": {
        "en": {
            "label": "Data Directory",
            "info": "The directory where the dataset is stored"
        },
        "zh": {
            "label": "数据目录",
            "info": "存放数据集的目录"
        }
    },
    "save_dir": {
        "en": {
            "label": "Save Directory",
            "info": "The directory to save the output data"
        },
        "zh": {
            "label": "保存目录",
            "info": "保存输出数据的目录"
        }
    },
    "save_intermediate_data": {
        "en": {
            "label": "Save Intermediate Data",
            "info": "Whether to save intermediate processing results"
        },
        "zh": {
            "label": "保存中间数据",
            "info": "是否保存中间处理结果"
        }
    },
    "save_note": {
        "en": {
            "label": "Save Note",
            "info": "Optional notes or descriptions for the save process"
        },
        "zh": {
            "label": "保存备注",
            "info": "保存过程的可选备注或描述"
        }
    },
    "seed": {
        "en": {
            "label": "Random Seed",
            "info": "The seed value for random number generation to ensure reproducibility"
        },
        "zh": {
            "label": "随机种子",
            "info": "用于随机数生成的种子值，以确保可重复性"
        }
    },
    "dataset_name": {
        "en": {
            "label": "Dataset Name",
            "info": "The name of the dataset to be used"
        },
        "zh": {
            "label": "数据集名称",
            "info": "使用的数据集名称"
        }
    },
    "test_sample_num": {
        "en": {
            "label": "Number of Test Samples",
            "info": "The number of test samples to evaluate the model"
        },
        "zh": {
            "label": "测试样本数量",
            "info": "用于评估模型的测试样本数量"
        }
    },
    "random_sample": {
        "en": {
            "label": "Random Sampling",
            "info": "Whether to use random sampling for test samples (only valid in test and dev set)"
        },
        "zh": {
            "label": "随机采样",
            "info": "是否对测试样本使用随机采样 (仅在dev和test集下生效)"
        }
    },
    "evaluate_preview_btn": {
        "en": {
            "value": "Preview evaluate configs",
        },
        "zh": {
            "value": "预览评估配置",
        }
    },
    "evaluate_run_btn": {
        "en": {
            "value": "Run Evaluate",
        },
        "zh": {
            "value": "运行评估实验",
        }
    },
    "use_metrics": {
        "en": {
            "label": "Evaluate Metrics",
            "info": "Evaluate Metrics used for evaluation"
        },
        "zh": {
            "label": "评价指标",
            "info": "最终使用的评价指标(可多选)"
        }
    },
    "save_metric_score": {
        "en": {
            "label": "Save Metric Score",
            "info": "Whether to save the metric score for evaluation purposes"
        },
        "zh": {
            "label": "保存指标分数",
            "info": "是否保存评估用的指标分数"
        }
    },
    "terminal_info": {
        "en": {
            "value": "<h4><center>FlashRAG Terminal</center></h4>"
        },
        "zh": {
            "value": "<h4><center>FlashRAG 终端</center></h4>"
        }
    },
    "terminal": {
        'en': {
            "label": "Terminal",
            "value": "Ready."
        },
        "zh": {
            "label": "终端",
            "value": "就绪."
        }
    },
    "output_box": {
        "en": {
            "value": "Ready."
        },
        "zh": {
            "value": "就绪。"
        },
    },
    "query": {
        "en": {
            "placeholder": "Input your query here..."
        },
        "zh": {
            "placeholder": "在这里输入你的问题..."
        }
    },
    "chatbot": {
        "en": {
            "label": "RAG chat board"
        },
        "zh": {
            "label": "RAG聊天面板"
        }
    },
    "system" : {
        "en": {
            "label": "System Prompt",
            "placeholder": "Input your system prompt here..."
        },
        "zh": {
            "label": "系统提示",
            "placeholder": "在这里输入你的系统提示..."
        }
    },
    "llmlingua_info" : {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression</h3>
  <p><strong>Authors:</strong> Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu</p>
  <p><strong>Abstract:</strong> In long context scenarios, large language models (LLMs) face three main challenges: higher computational cost, performance reduction, and position bias. Research indicates that LLM performance hinges on the density and position of key information in the input prompt. Inspired by these findings, LongLLMLingua is proposed for prompt compression to improve LLMs' perception of key information and simultaneously address these challenges. Extensive evaluation across various scenarios demonstrates that LongLLMLingua enhances performance, reduces costs, and minimizes latency. For instance, it improves performance by up to 21.4% in the NaturalQuestions benchmark with approximately 4x fewer tokens, achieving a 94.0% cost reduction in the LooGLE benchmark. Additionally, it accelerates end-to-end latency by 1.4x-2.6x when compressing prompts at 2x-6x ratios. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2310.06839" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">LongLLMLingua: 基于提示压缩的长上下文场景下LLM的加速与性能提升</h3>
  <p><strong>作者：</strong> Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu</p>
  <p><strong>摘要：</strong> 在长上下文场景中，大型语言模型（LLM）面临三个主要挑战：较高的计算成本、性能下降以及位置偏差。研究表明，LLM 的性能与输入提示中关键信息的密度和位置密切相关。受此启发，提出了 LongLLMLingua 以进行提示压缩，提升 LLM 对关键信息的感知，解决上述挑战。广泛的实验表明，LongLLMLingua 不仅提升了性能，还显著降低了成本和延迟。例如，在 NaturalQuestions 基准测试中，LongLLMLingua 将性能提高了最多 21.4%，同时减少了约 4 倍的 token 数，在 LooGLE 基准测试中实现了 94.0% 的成本降低。此外，当以 2 倍至 6 倍的压缩比压缩提示时，LongLLMLingua 可将端到端延迟加速 1.4 倍至 2.6 倍。<br><strong>论文：</strong>详见 <a href="https://arxiv.org/abs/2310.06839" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "recomp_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation</h3>
  <p><strong>Authors:</strong> Fangyuan Xu, Weijia Shi, Eunsol Choi</p>
  <p><strong>Abstract:</strong> Retrieving documents and prepending them in-context at inference time improves performance of language model (LMs) on a wide range of tasks. However, these documents, often spanning hundreds of words, make inference substantially more expensive. We propose compressing the retrieved documents into textual summaries prior to in-context integration. This not only reduces the computational costs but also relieves the burden of LMs to identify relevant information in long retrieved documents. We present two compressors—an extractive compressor which selects useful sentences from retrieved documents, and an abstractive compressor which generates summaries by synthesizing information from multiple documents. Both compressors are trained to improve LMs' performance on end tasks when the generated summaries are prepended to the LMs' input, while keeping the summary relevant. If the retrieved documents are irrelevant to the input or offer no additional information to the LM, our compressor can return an empty string, implementing selective augmentation. We evaluate our approach on language modeling tasks and open-domain question answering tasks. We achieve a compression rate as low as 6% with minimal loss in performance for both tasks, significantly outperforming off-the-shelf summarization models. We show that our compressors trained for one LM can transfer to other LMs on the language modeling task and provide summaries largely faithful to the retrieved documents. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2310.04408" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">RECOMP: 基于压缩和选择性增强的检索增强型语言模型改进方法</h3>
  <p><strong>作者：</strong> Fangyuan Xu, Weijia Shi, Eunsol Choi</p>
  <p><strong>摘要：</strong> 在推理时检索文档并将其作为上下文预先加入，可以提升语言模型（LM）在各种任务上的性能。然而，这些文档通常跨越数百个单词，使得推理开销大大增加。我们提出在将检索文档集成到上下文中之前，将其压缩为文本摘要。这不仅减少了计算成本，还减轻了语言模型在长文档中识别相关信息的负担。我们提出了两种压缩方法——抽取式压缩器，从检索的文档中选择有用的句子；和生成式压缩器，通过综合多文档的信息生成摘要。两种压缩器都经过训练，在将生成的摘要作为输入添加到语言模型中时，能够提升任务的性能，并确保摘要的相关性。如果检索的文档与输入无关或未提供有价值的信息，我们的压缩器可以返回空字符串，实现选择性增强。我们在语言建模任务和开放域问答任务上评估了我们的方法。我们的压缩率最低可达 6%，且对两项任务的性能损失最小，显著优于现成的摘要模型。我们还展示了为一种语言模型训练的压缩器可以迁移到其他语言模型上，且生成的摘要对检索文档的忠实度较高。<br><strong>论文：</strong>详见 <a href="https://arxiv.org/abs/2310.04408" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "sc_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">Compressing Context to Enhance Inference Efficiency of Large Language Models</h3>
  <p><strong>Authors:</strong> Yucheng Li, Bo Dong, Chenghua Lin, Frank Guerin</p>
  <p><strong>Abstract:</strong> Large language models (LLMs) have achieved remarkable performance across various tasks. However, they face challenges in managing long documents and extended conversations, due to significantly increased computational requirements, both in memory and inference time, and potential context truncation when the input exceeds the LLM's fixed context length. This paper proposes a method called Selective Context that enhances the inference efficiency of LLMs by identifying and pruning redundancy in the input context to make the input more compact. We test our approach using common data sources requiring long context processing: arXiv papers, news articles, and long conversations, on tasks of summarization, question answering, and response generation. Experimental results show that Selective Context significantly reduces memory cost and decreases generation latency while maintaining comparable performance to that achieved when full context is used. Specifically, we achieve a 50% reduction in context cost, resulting in a 36% reduction in inference memory usage and a 32% reduction in inference time, while observing only a minor drop of 0.023 in BERTscore and 0.038 in faithfulness on four downstream applications, indicating that our method strikes a good balance between efficiency and performance. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2310.06201" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">压缩上下文以提升大型语言模型的推理效率</h3>
  <p><strong>作者：</strong> Yucheng Li, Bo Dong, Chenghua Lin, Frank Guerin</p>
  <p><strong>摘要：</strong> 大型语言模型（LLM）在各种任务上取得了显著的成绩。然而，它们在处理长文档和扩展对话时面临挑战，主要由于计算需求的显著增加，包括内存和推理时间，并且当输入超过 LLM 的固定上下文长度时，还可能会出现上下文截断问题。本文提出了一种名为 Selective Context（选择性上下文）的方法，通过识别并修剪输入上下文中的冗余部分，使输入更加紧凑，从而提升 LLM 的推理效率。我们在需要长上下文处理的常见数据源上进行测试，包括 arXiv 论文、新闻文章和长对话，任务包括摘要生成、问答和响应生成。实验结果表明，Selective Context 显著降低了内存成本，并减少了生成延迟，同时在性能上与使用完整上下文时相当。具体而言，我们在上下文成本上实现了 50% 的减少，推理内存使用量减少了 36%，推理时间减少了 32%，同时在四个下游应用中，BERTscore 的下降仅为 0.023，忠实度下降为 0.038，表明该方法在效率与性能之间取得了良好的平衡。<br><strong>论文：</strong> 详见 <a href="https://arxiv.org/abs/2310.06201" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "retrobust_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">Making Retrieval-Augmented Language Models Robust to Irrelevant Context</h3>
  <p><strong>Authors:</strong> Ori Yoran, Tomer Wolfson, Ori Ram, Jonathan Berant</p>
  <p><strong>Abstract:</strong> Retrieval-augmented language models (RALMs) hold promise to produce language understanding systems that are factual, efficient, and up-to-date. An important desideratum of RALMs is that retrieved information helps model performance when it is relevant, and does not harm performance when it is not. This is particularly important in multi-hop reasoning scenarios, where misuse of irrelevant evidence can lead to cascading errors. However, recent work has shown that retrieval augmentation can sometimes have a negative effect on performance. In this work, we present a thorough analysis on five open-domain question answering benchmarks, characterizing cases when retrieval reduces accuracy. We then propose two methods to mitigate this issue. First, a simple baseline that filters out retrieved passages that do not entail question-answer pairs according to a natural language inference (NLI) model. This is effective in preventing performance reduction, but at a cost of also discarding relevant passages. Thus, we propose a method for automatically generating data to fine-tune the language model to properly leverage retrieved passages, using a mix of relevant and irrelevant contexts at training time. We empirically show that even 1,000 examples suffice to train the model to be robust to irrelevant contexts while maintaining high performance on examples with relevant ones. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2310.01558" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">增强型检索语言模型的抗干扰能力：应对无关上下文的挑战</h3>
  <p><strong>作者：</strong> Ori Yoran, Tomer Wolfson, Ori Ram, Jonathan Berant</p>
  <p><strong>摘要：</strong> 增强型检索语言模型（RALMs）有望创建既准确高效又实时更新的语言理解系统。RALMs 的一个重要特性是，当检索的信息与任务相关时能提升模型性能，而当信息无关时不应损害模型的性能。这在多跳推理任务中尤为重要，因为无关证据的错误使用可能导致连锁错误。然而，最近的研究表明，检索增强有时会对性能产生负面影响。在本研究中，我们对五个开放领域的问答基准进行了详细分析，分析了检索信息何时会导致准确率下降。随后，我们提出了两种方法来缓解这一问题。首先是一个简单的基线方法，通过自然语言推理（NLI）模型过滤掉不包含问答对的检索段落。此方法能够有效防止性能下降，但也会丢弃一些相关段落。因此，我们提出了一种方法，利用相关和无关上下文的混合数据，在训练阶段自动生成数据来微调语言模型，以便更好地利用检索到的段落。我们通过实验证明，即使只有 1,000 个样本，也足以训练模型在处理无关上下文时保持鲁棒性，同时在处理相关上下文时保持高性能。<br><strong>论文：</strong> 详见 <a href="https://arxiv.org/abs/2310.01558" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "skr_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">Self-Knowledge Guided Retrieval Augmentation for Large Language Models</h3>
  <p><strong>Authors:</strong> Yile Wang, Peng Li, Maosong Sun, Yang Liu</p>
  <p><strong>Abstract:</strong> Large language models (LLMs) have shown superior performance without task-specific fine-tuning. Despite their success, the knowledge stored in the parameters of LLMs could still be incomplete and difficult to update due to computational costs. As a complementary approach, retrieval-based methods offer non-parametric world knowledge and improve performance on tasks like question answering. However, we find that retrieved knowledge does not always help and may even negatively impact original responses. To better leverage both internal knowledge and external world knowledge, we investigate eliciting the model's ability to recognize what they know and do not know (referred to as self-knowledge) and propose Self-Knowledge Guided Retrieval Augmentation (SKR). SKR is a simple yet effective method that enables LLMs to refer to previously encountered questions and adaptively call upon external resources when dealing with new questions. We evaluate SKR on multiple datasets and show that it outperforms chain-of-thought-based and fully retrieval-based methods when using either InstructGPT or ChatGPT. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2310.05002" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">自我知识引导的检索增强方法：提升大型语言模型表现</h3>
  <p><strong>作者：</strong> Yile Wang, Peng Li, Maosong Sun, Yang Liu</p>
  <p><strong>摘要：</strong> 大型语言模型（LLMs）在无需特定任务微调的情况下已表现出卓越的性能。尽管如此，LLMs 参数中存储的知识可能仍然不完整且难以更新，主要由于计算成本的问题。作为补充，基于检索的方法可以提供非参数化的世界知识，提升问答等任务的表现。然而，我们发现检索到的知识并不总是有效，甚至在某些情况下可能对原始回答产生负面影响。为了更好地利用内部知识和外部世界知识，我们研究了引导模型识别它们已知与未知知识的能力（即自我知识），并提出了自我知识引导的检索增强（SKR）。SKR 是一种简单而有效的方法，能够使 LLM 在处理新问题时参考其先前遇到的问题，并在需要时自适应调用外部资源。我们在多个数据集上评估了 SKR，并展示了其在使用 InstructGPT 或 ChatGPT 时，超越了基于链式推理和完全基于检索的方法。<br><strong>论文：</strong> 详见 <a href="https://arxiv.org/abs/2310.05002" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "selfrag_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection</h3>
  <p><strong>Authors:</strong> Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi</p>
  <p><strong>Abstract:</strong> Despite their remarkable capabilities, large language models (LLMs) often produce responses containing factual inaccuracies due to their sole reliance on the parametric knowledge they encapsulate. Retrieval-Augmented Generation (RAG), an approach that augments LMs with retrieval of relevant knowledge, helps to mitigate such issues. However, indiscriminately retrieving and incorporating a fixed number of passages, regardless of their relevance, diminishes LM versatility or can lead to unhelpful responses. We introduce a new framework called Self-Reflective Retrieval-Augmented Generation (Self-RAG), which enhances an LM's quality and factuality through retrieval and self-reflection. Our framework trains a single arbitrary LM that adaptively retrieves passages on demand, and generates and reflects on both retrieved passages and its own generations using special tokens, called reflection tokens. The generation of reflection tokens makes the LM controllable during inference, enabling it to adapt its behavior to diverse task requirements. Experiments show that Self-RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models across various tasks. Specifically, Self-RAG outperforms ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, reasoning, and fact verification tasks. It also shows substantial improvements in factuality and citation accuracy for long-form generations compared to these models. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2310.11511" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">自我反思引导的检索增强生成方法（Self-RAG）</h3>
  <p><strong>作者：</strong> Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi</p>
  <p><strong>摘要：</strong> 尽管大型语言模型（LLMs）具有出色的能力，但它们常常因仅依赖其封装的参数化知识而产生包含事实错误的回答。检索增强生成（RAG）是一种通过检索相关知识增强语言模型的方法，有助于减少这些问题。然而，盲目地检索并结合固定数量的段落，不论其相关性如何，都会削弱语言模型的多样性，甚至导致生成无用的回答。我们提出了一种新的框架——自我反思引导的检索增强生成（Self-RAG），通过检索和自我反思提升语言模型的质量和事实性。我们的框架训练一个可以根据需求自适应检索段落的语言模型，并使用特殊的“反思标记”生成和反思检索到的段落及其自身生成的内容。生成反思标记使得模型在推理过程中变得可控，能够根据不同任务要求调整行为。实验表明，Self-RAG（7B 和 13B 参数）在多项任务上显著超越了当前最先进的语言模型和检索增强模型。具体而言，Self-RAG 在开放域问答、推理和事实验证任务中超越了 ChatGPT 和检索增强版 Llama2-chat，并在长文本生成的事实性和引用准确性方面表现出显著的提升。<br><strong>论文：</strong> 详见 <a href="https://arxiv.org/abs/2310.11511" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "flare_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">Active Retrieval Augmented Generation</h3>
  <p><strong>Authors:</strong> Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig</p>
  <p><strong>Abstract:</strong> Despite the remarkable ability of large language models (LMs) to comprehend and generate language, they often tend to hallucinate and produce factually inaccurate output. Augmenting LMs by retrieving information from external knowledge resources is a promising solution. However, most existing retrieval-augmented LMs use a retrieve-and-generate setup that only retrieves information once based on the input, which is limiting in more general scenarios involving the generation of long texts, where continuously gathering information throughout the generation process is essential. In this work, we present a generalized view of active retrieval-augmented generation methods, which actively decide when and what to retrieve across the course of generation. We propose Forward-Looking Active REtrieval Augmented Generation (FLARE), a generic method that iteratively predicts the upcoming sentence, anticipates future content, and uses it as a query to retrieve relevant documents to regenerate the sentence if it contains low-confidence tokens. We evaluate FLARE extensively along with baselines over 4 long-form knowledge-intensive generation tasks and datasets. FLARE achieves superior or competitive performance on all tasks, demonstrating the effectiveness of our method. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2305.06983" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">主动检索增强生成方法（Active Retrieval Augmented Generation）</h3>
  <p><strong>作者：</strong> Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig</p>
  <p><strong>摘要：</strong> 尽管大型语言模型（LLMs）在理解和生成语言方面表现出色，但它们常常出现幻觉，并生成事实不准确的输出。通过从外部知识资源中检索信息来增强语言模型是一个有前景的解决方案。然而，大多数现有的检索增强语言模型仅使用检索和生成的模式，在输入的基础上只检索一次信息，这在涉及生成长文本的更一般化场景中是有限的，因为在生成过程中持续收集信息至关重要。本文提出了一种主动检索增强生成（Active Retrieval Augmented Generation，ARAG）方法，这些方法在生成过程中主动决定何时以及检索什么信息。我们提出了前瞻性主动检索增强生成（FLARE）方法，它通过预测即将生成的句子，预见未来的内容，并将其作为查询检索相关文档，如果生成的句子包含低置信度的标记，则重新生成该句子。我们在四个长文本知识密集型生成任务和数据集上全面测试了FLARE及基准方法，结果显示FLARE在所有任务中都取得了优异或具有竞争力的表现，证明了我们方法的有效性。<br><strong>论文：</strong> 详见 <a href="https://arxiv.org/abs/2305.06983" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "iterretgen_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy</h3>
  <p><strong>Authors:</strong> Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, Weizhu Chen</p>
  <p><strong>Abstract:</strong> Large language models are powerful text processors and reasoners, but are still subject to limitations such as outdated knowledge and hallucinations, necessitating a connection to the world. Retrieval-augmented large language models (RALMs) have gained attention for grounding model generation on external knowledge. However, retrievers often struggle to capture relevance, especially for queries with complex information needs. Recent work has proposed improving relevance modeling by involving large language models in the retrieval process, i.e., improving retrieval with generation. In this paper, we present Iter-RetGen, a method that synergizes retrieval and generation in an iterative manner. The model's output provides insights into what might be needed to complete a task, and this informs the retrieval of more relevant knowledge, which, in turn, helps generate a better output in the next iteration. Unlike recent approaches that interleave retrieval with generation, Iter-RetGen processes all retrieved knowledge as a whole, maintaining flexibility in generation without structural constraints. We evaluate Iter-RetGen on multi-hop question answering, fact verification, and commonsense reasoning tasks. Our results show that it can effectively leverage both parametric and non-parametric knowledge, outperforming or competing with state-of-the-art retrieval-augmented baselines while reducing retrieval and generation overhead. Furthermore, performance can be enhanced through generation-augmented retrieval adaptation. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2305.15294" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">通过迭代检索-生成协同增强检索增强大型语言模型</h3>
  <p><strong>作者：</strong> Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, Weizhu Chen</p>
  <p><strong>摘要：</strong> 大型语言模型（LLMs）是强大的文本处理器和推理器，但仍然面临过时知识和幻觉等限制，这要求它们与外部世界连接。检索增强大型语言模型（RALMs）通过将外部知识作为生成基础，获得了广泛关注。然而，现有的检索方法在捕捉相关性方面存在困难，尤其是在处理复杂信息需求的查询时。最近的研究提出了通过让大型语言模型参与检索来改善相关性建模，即通过生成来改进检索。本文提出了一种名为Iter-RetGen的方法，它通过迭代方式将检索与生成协同作用。模型的输出提供了完成任务所需的信息，从而为检索更多相关知识提供了有益的上下文，这些知识又有助于在下一次迭代中生成更好的输出。与最近的生成-检索交替方法不同，Iter-RetGen以整体方式处理所有检索到的知识，在不增加结构性约束的情况下，保持生成过程的灵活性。我们在多跳问答、事实验证和常识推理等任务上评估了Iter-RetGen。实验结果表明，它能够有效地利用参数化和非参数化的知识，优于或与现有的检索增强基线方法竞争，同时减少了检索和生成的开销。我们还可以通过生成增强的检索适配进一步提高性能。<br><strong>论文：</strong> 详见 <a href="https://arxiv.org/abs/2305.15294" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "ircot_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions</h3>
  <p><strong>Authors:</strong> Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal</p>
  <p><strong>Abstract:</strong> Prompting-based large language models (LLMs) have demonstrated remarkable ability to generate natural language reasoning steps or Chains-of-Thoughts (CoT) for multi-step question answering (QA). However, they struggle when the necessary knowledge is either unavailable or outdated within the model’s parameters. While using the question to retrieve relevant text from an external knowledge source helps LLMs, we observe that the one-step retrieve-and-read approach is insufficient for multi-step QA. In such scenarios, what to retrieve depends on what has already been derived, which in turn may depend on what was previously retrieved. To address this, we propose IRCoT, a new approach for multi-step QA that interleaves retrieval with steps (sentences) in a CoT, guiding retrieval with CoT and using retrieved results to improve CoT. We demonstrate that using IRCoT with GPT-3 significantly improves retrieval (up to 21 points) and downstream QA (up to 15 points) on four datasets: HotpotQA, 2WikiMultihopQA, MuSiQue, and IIRC. These improvements are observed in out-of-distribution (OOD) settings and with smaller models like Flan-T5-large without additional training. IRCoT reduces model hallucinations, resulting in factually more accurate CoT reasoning. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2212.10509" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">将检索与链式思维推理交替应用于知识密集型多步骤问题</h3>
  <p><strong>作者：</strong> Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal</p>
  <p><strong>摘要：</strong> 基于提示的大型语言模型（LLMs）在多步骤问答（QA）中表现出了强大的能力，能够生成自然语言推理步骤或链式思维（Chain-of-Thoughts，CoT）。然而，当所需的知识在模型参数中不可用或过时时，LLM会遇到困难。虽然通过使用问题从外部知识源检索相关文本可以帮助LLM，但我们观察到这种一步检索-读取的方法不足以解决多步骤问答的问题。在这种场景下，检索内容的选择依赖于已经推导出的内容，而这些内容可能又依赖于之前检索到的信息。为了解决这个问题，我们提出了IRCoT，一种新的多步骤问答方法，它将检索与链式思维中的步骤（句子）交替进行，通过链式思维引导检索，并利用检索到的结果改进链式思维。我们展示了使用IRCoT与GPT-3结合可以显著提高检索（最多提高21分）和下游问答（最多提高15分）在四个数据集上的表现：HotpotQA、2WikiMultihopQA、MuSiQue和IIRC。我们还在分布外（OOD）设置和不进行额外训练的小型模型（如Flan-T5-large）中观察到了类似的显著提升。IRCoT减少了模型的幻觉现象，从而使链式推理更加准确。<br><strong>论文：</strong> 详见 <a href="https://arxiv.org/abs/2212.10509" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "trace_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">TRACE the Evidence: Constructing Knowledge-Grounded Reasoning Chains for Retrieval-Augmented Generation</h3>
  <p><strong>Authors:</strong> Jinyuan Fang, Zaiqiao Meng, Craig Macdonald</p>
  <p><strong>Abstract:</strong> Retrieval-augmented generation (RAG) has proven to be an effective approach for addressing question answering (QA) tasks. However, the imperfections of retrievers in RAG models often lead to the retrieval of irrelevant information, which can introduce noise and degrade performance, especially when handling multi-hop questions that require multiple reasoning steps. To improve the multi-hop reasoning ability of RAG models, we propose TRACE, a method that constructs knowledge-grounded reasoning chains. These chains consist of a series of logically connected knowledge triples that help identify and integrate supporting evidence from retrieved documents for answering questions. Specifically, TRACE uses a Knowledge Graph (KG) Generator to create a knowledge graph from the retrieved documents and an Autoregressive Reasoning Chain Constructor to build the reasoning chains. Experimental results on three multi-hop QA datasets show that TRACE achieves an average performance improvement of up to 14.03% compared to using all the retrieved documents. Furthermore, the results indicate that using reasoning chains as context, rather than the entire documents, is often sufficient to correctly answer questions. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2406.11460" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">TRACE证据：构建面向检索增强生成的知识驱动推理链</h3>
  <p><strong>作者：</strong> Jinyuan Fang, Zaiqiao Meng, Craig Macdonald</p>
  <p><strong>摘要：</strong> 检索增强生成（RAG）已被证明是解决问答（QA）任务的有效方法。然而，RAG模型中检索器的不完美常常导致检索到无关的信息，这可能引入噪声并降低性能，尤其是在处理需要多步推理的多跳问题时。为了提升RAG模型的多跳推理能力，我们提出了TRACE方法。TRACE构建了知识驱动的推理链，这些推理链是由一系列逻辑连接的知识三元组组成，帮助从检索到的文档中识别并整合支持证据以回答问题。具体而言，TRACE使用知识图（KG）生成器从检索到的文档中创建知识图，然后使用自回归推理链构建器来构建推理链。在三个多跳问答数据集上的实验结果表明，与使用所有检索到的文档相比，TRACE在性能上平均提高了14.03%。此外，结果表明，使用推理链作为上下文，而不是整个文档，通常足以正确回答问题。<br><strong>论文：</strong> 详见 <a href="https://arxiv.org/abs/2406.11460" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "spring_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">One Token Can Help! Learning Scalable and Pluggable Virtual Tokens for Retrieval-Augmented Large Language Models</h3>
  <p><strong>Authors:</strong> Yutao Zhu, Zhaoheng Huang, Zhicheng Dou, Ji-Rong Wen</p>
  <p><strong>Abstract:</strong> Retrieval-augmented generation (RAG) is a promising approach for improving large language models (LLMs) to generate more factual, accurate, and up-to-date content. Existing methods either optimize prompts to guide LLMs in utilizing retrieved information or fine-tune LLMs to adapt to RAG scenarios. Although fine-tuning can enhance performance, it often compromises the general generation capabilities of LLMs by modifying their parameters. This limitation poses challenges in practical applications, especially for deployed LLMs, as parameter adjustments may interfere with their original functionality. To overcome this, we propose a novel method of learning scalable and pluggable virtual tokens for RAG. By preserving the LLMs' original parameters and fine-tuning only the embeddings of these pluggable tokens, our approach improves LLMs' performance while maintaining their general generation capabilities. Additionally, we design training strategies to enhance the scalability, flexibility, and generalizability of our method. Extensive experiments across 12 question-answering tasks demonstrate the effectiveness and superiority of our approach. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2405.19670" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">一枚虚拟Token能帮忙！学习可扩展且可插拔的虚拟Token以增强检索增强型大语言模型</h3>
  <p><strong>作者：</strong> Yutao Zhu, Zhaoheng Huang, Zhicheng Dou, Ji-Rong Wen</p>
  <p><strong>摘要：</strong> 检索增强生成（RAG）是提升大语言模型（LLM）生成更为准确、真实和更新内容的有前景的方式。现有方法要么通过优化提示词引导LLMs使用检索到的信息，要么直接对LLMs进行微调以适应RAG场景。虽然微调可以提高性能，但通常会通过修改LLM的参数，影响其原本的生成能力。这种局限性在实际应用中尤为重要，特别是在LLM已经部署的情况下，因为参数的调整可能会影响其原有的功能。为了解决这个问题，我们提出了一种新方法，学习可扩展且可插拔的虚拟Token用于RAG。通过保留LLM的原始参数，仅微调这些可插拔Token的嵌入，我们的方法不仅提高了LLM的性能，同时保持了其通用生成能力。此外，我们设计了多种训练策略，以增强我们方法的可扩展性、灵活性和泛化能力。通过在12个问答任务中的综合实验，我们证明了该方法的优越性。<br><strong>论文：</strong> 详见 <a href="https://arxiv.org/abs/2405.19670" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "adaptive_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity</h3>
  <p><strong>Authors:</strong> Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park</p>
  <p><strong>Abstract:</strong> Retrieval-Augmented Large Language Models (LLMs), which incorporate non-parametric knowledge from external knowledge bases into LLMs, have emerged as a promising approach to enhancing response accuracy in various tasks such as Question-Answering (QA). However, although there are various methods for handling queries of different complexities, they either process simple queries with unnecessary computational overhead or fail to properly address complex, multi-step queries. Yet, not all user requests fit neatly into simple or complex categories. In this work, we propose a novel adaptive QA framework that dynamically selects the most suitable strategy for retrieval-augmented LLMs, from the simplest to the most sophisticated, based on the query complexity. This selection process is operationalized using a classifier, a smaller LM trained to predict the complexity level of incoming queries with labels automatically collected from predicted outcomes and inductive biases in datasets. Our approach offers a balanced strategy, seamlessly adapting between iterative and single-step retrieval-augmented LLMs, as well as no-retrieval methods, depending on the complexity of the query. We validate our model on a range of open-domain QA datasets with varying query complexities, showing that it enhances both efficiency and accuracy of QA systems compared to relevant baselines, including adaptive retrieval approaches. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2403.14403" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">Adaptive-RAG: 通过问题复杂度学习适应检索增强型大语言模型</h3>
  <p><strong>作者：</strong> Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park</p>
  <p><strong>摘要：</strong> 检索增强型大语言模型（LLMs）通过结合外部知识库的非参数化知识，提高了在问答（QA）等任务中的响应准确性。尽管已有多种方法处理不同复杂度的查询，但它们要么以不必要的计算开销处理简单查询，要么未能充分处理复杂的多步骤查询；然而，并非所有用户请求都可以简单地划分为简单或复杂的类别。在这项工作中，我们提出了一种新的自适应QA框架，可以根据查询复杂度动态选择最合适的策略，从最简单到最复杂的检索增强型LLM策略。该选择过程通过一个分类器实现，该分类器是一个小型语言模型，经过训练能够根据自动收集的标签预测传入查询的复杂度，标签来自于模型的预测结果和数据集中的内在归纳偏差。我们的方法提供了一种平衡的策略，能够根据查询的复杂性，在迭代和单步骤检索增强型LLM之间以及无检索方法之间无缝切换。我们在多个开放领域QA数据集上验证了该模型，涵盖了不同复杂度的查询，并表明，相较于相关基线方法，包括自适应检索方法，我们的方法提高了问答系统的整体效率和准确性。<br><strong>论文：</strong> 详见 <a href="https://arxiv.org/abs/2403.14403" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "rqrag_info": {
        "en": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation</h3>
  <p><strong>Authors:</strong> Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo, Wei Xue, Yike Guo, Jie Fu</p>
  <p><strong>Abstract:</strong> Large Language Models (LLMs) exhibit remarkable capabilities but are prone to generating inaccurate or hallucinatory responses. This limitation stems from their reliance on vast pretraining datasets, making them susceptible to errors in unseen scenarios. To tackle these challenges, Retrieval-Augmented Generation (RAG) addresses this by incorporating external, relevant documents into the response generation process, thus leveraging non-parametric knowledge alongside LLMs' in-context learning abilities. However, existing RAG implementations primarily focus on initial input for context retrieval, overlooking the nuances of ambiguous or complex queries that necessitate further clarification or decomposition for accurate responses. In this work, we propose learning to Refine Query for Retrieval Augmented Generation (RQ-RAG), equipping the model with explicit rewriting, decomposition, and disambiguation capabilities to enhance its performance. Our experimental results show that RQ-RAG applied to a 7B Llama2 model surpasses the previous state-of-the-art (SOTA) by an average of 1.9% across three single-hop QA datasets, and also demonstrates enhanced performance in handling complex, multi-hop QA tasks. <br><strong>Paper:</strong> Available at <a href="https://arxiv.org/abs/2404.00610" target="_blank">arXiv</a>.</p>
</div>"""
        },
        "zh": {
            "value": """<div class="method-card" style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px;">
  <h3 style="color: #333;">RQ-RAG: 学习为检索增强生成模型优化查询</h3>
  <p><strong>作者：</strong> Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo, Wei Xue, Yike Guo, Jie Fu</p>
  <p><strong>摘要：</strong> 大语言模型（LLMs）具有出色的能力，但容易生成不准确或虚构的响应。这一局限性源自于它们对庞大预训练数据集的依赖，使得它们在遇到未知场景时容易出错。为了解决这些挑战，检索增强生成（RAG）通过将外部相关文档融入生成过程中，利用非参数化知识与LLM的上下文学习能力。然而，现有的RAG实现主要侧重于输入上下文的检索，而忽视了对模糊或复杂查询的处理，这些查询通常需要进一步的澄清或分解，以产生准确的回答。为此，本文提出了为检索增强生成（RQ-RAG）学习优化查询的方法，通过显式的重写、分解和消歧义能力，提升模型的表现。实验结果表明，当RQ-RAG应用于7B Llama2模型时，在三个单跳问答数据集上平均超越了先前的最先进方法（SOTA）1.9%，并且在处理复杂的多跳问答任务时也表现出色。<br><strong>论文：</strong> 详见 <a href="https://arxiv.org/abs/2404.00610" target="_blank">arXiv</a>。</p>
</div>"""
        }
    },
    "config_file": {
        "en": {
            "label": "Config File",
        },
        "zh": {
            "label": "配置文件",
        }
    },
    "llmlingua_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "recomp_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "sc_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "retrobust_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "skr_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "selfrag_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "flare_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "iterretgen_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "ircot_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "trace_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "spring_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "adaptive_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "rqrag_advanced": {
        "en": {
            "label": "settings",
        },
        "zh": {
            "label": "参数设置",
        }
    },
    "evaluate_output_box": {
        "en": {
            "value": "Ready."
        },
        "zh": {
            "value": "就绪。"
        }
    },
    "chat_tab": {
        "en": {
            "label": "Chat"
        },
        "zh": {
            "label": "聊天"
        }
    },
    "evaluate_tab": {
        "en": {
            "label": "Evaluate"
        },
        "zh": {
            "label": "评估"
        }
    },
    "dataset_split": {
        "en": {
            "label": "Dataset Split",
            "info": "The specify dataset split used for evaluation"
        },
        "zh": {
            "label": "数据集划分",
            "info": "评估时使用的数据集划分"
        }
    }
}