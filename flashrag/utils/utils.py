import os
import importlib
from transformers import AutoConfig
from flashrag.dataset.dataset import Dataset

def get_dataset(config):
    """Load dataset from config."""

    dataset_path = config['dataset_path']
    all_split = config['split']

    split_dict = {split: None for split in all_split}

    for split in all_split:
        split_path = os.path.join(dataset_path, f'{split}.jsonl')
        if not os.path.exists(split_path):
            print(f"{split} file not exists!")
            continue
        if split in ['test','val','dev']:
            split_dict[split] = Dataset(config,
                                        split_path,
                                        sample_num = config['test_sample_num'],
                                        random_sample = config['random_sample'])
        else:
            split_dict[split] = Dataset(config, split_path)

    return split_dict

def get_generator(config, **params):
    """Automatically select generator class based on config."""
    if config['framework'] == 'vllm':
        return getattr(
                importlib.import_module("flashrag.generator"),
                "VLLMGenerator"
            )(config, **params)
    elif config['framework'] == 'fschat':
        return getattr(
                importlib.import_module("flashrag.generator"),
                "FastChatGenerator"
            )(config, **params)
    elif config['framework'] == 'hf':
        if "t5" in config['generator_model'] or "bart" in config['generator_model']:
            return getattr(
                importlib.import_module("flashrag.generator"),
                "EncoderDecoderGenerator"
            )(config, **params)
        else:
            return getattr(
                    importlib.import_module("flashrag.generator"),
                    "HFCausalLMGenerator"
                )(config, **params)
    elif config['framework'] == 'openai':
        return getattr(
                    importlib.import_module("flashrag.generator"),
                    "OpenaiGenerator"
                )(config, **params)
    else:
        raise NotImplementedError


def get_retriever(config):
    r"""Automatically select retriever class based on config's retrieval method

    Args:
        config (dict): configuration with 'retrieval_method' key

    Returns:
        Retriever: retriever instance
    """
    if config['retrieval_method'] == "bm25":
        return getattr(
            importlib.import_module("flashrag.retriever"),
            "BM25Retriever"
        )(config)
    else:
        return getattr(
            importlib.import_module("flashrag.retriever"),
            "DenseRetriever"
        )(config)

def get_reranker(config):
    model_path = config['rerank_model_path']
    # get model config
    model_config = AutoConfig.from_pretrained(model_path)
    arch = model_config.architectures[0]
    if 'forsequenceclassification' in arch.lower():
        return getattr(
            importlib.import_module("flashrag.retriever"),
            "CrossReranker"
        )(config)
    else:
        return getattr(
            importlib.import_module("flashrag.retriever"),
            "BiReranker"
        )(config)


def get_judger(config):
    judger_name = config['judger_name']
    if 'skr' in judger_name.lower():
        return getattr(
                importlib.import_module("flashrag.judger"),
                "SKRJudger"
            )(config)
    else:
        assert False, "No implementation!"

def get_refiner(config):
    refiner_name = config['refiner_name']
    refiner_path = config['refiner_model_path']

    default_path_dict = {
        'recomp_abstractive_nq': 'fangyuan/nq_abstractive_compressor',
        'recomp:abstractive_tqa': 'fangyuan/tqa_abstractive_compressor',
        'recomp:abstractive_hotpotqa': 'fangyuan/hotpotqa_abstractive',
    }

    if refiner_path is None:
        if refiner_name in default_path_dict:
            refiner_path = default_path_dict[refiner_name]
        else:
            assert False, "refiner_model_path is empty!"

    model_config = AutoConfig.from_pretrained(refiner_path)
    arch = model_config.architectures[0].lower()
    if "recomp" in refiner_name.lower() or \
        "recomp" in refiner_path or \
        'bert' in arch:
        if model_config.model_type == "t5" :
            return getattr(
                importlib.import_module("flashrag.refiner"),
                "AbstractiveRecompRefiner"
            )(config)
        else:
            return getattr(
                importlib.import_module("flashrag.refiner"),
                "ExtractiveRefiner"
            )(config)
    elif "lingua" in refiner_name.lower():
        return getattr(
                importlib.import_module("flashrag.refiner"),
                "LLMLinguaRefiner"
            )(config)
    elif "selective-context" in refiner_name.lower() or "sc" in refiner_name.lower():
        return getattr(
                importlib.import_module("flashrag.refiner"),
                "SelectiveContextRefiner"
            )(config)
    else:
        assert False, "No implementation!"