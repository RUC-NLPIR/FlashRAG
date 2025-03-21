import os
import re
import json
import importlib
from transformers import AutoConfig
from flashrag.dataset.dataset import Dataset


def get_dataset(config):
    """Load dataset from config."""
    SUPPORT_FILES = ["jsonl", "json", "parquet"]

    dataset_path = config["dataset_path"]
    all_split = config["split"]

    split_dict = {split: None for split in all_split}

    for split in all_split:
        exist_flag = 0
        for file_postfix in SUPPORT_FILES:
            split_path = os.path.join(dataset_path, f"{split}.{file_postfix}")
            if not os.path.exists(split_path):
                continue
            else:
                exist_flag = 1
                break
        if exist_flag == 0:
            continue
        else:
            print(f"Loading {split} dataset from: {split_path}...")
        if split in ["test", "val", "dev"]:
            split_dict[split] = Dataset(
                config, split_path, sample_num=config["test_sample_num"], random_sample=config["random_sample"]
            )
        else:
            split_dict[split] = Dataset(config, split_path)

    return split_dict


def get_generator(config, **params):
    """Automatically select generator class based on config."""

    if config['framework'] == 'openai':
        return getattr(importlib.import_module("flashrag.generator"), "OpenaiGenerator")(config, **params)
    
    # judge multimodal model
    with open(os.path.join(config["generator_model_path"], "config.json"), "r") as f:
        model_config = json.load(f)
    arch = model_config['architectures'][0]
    if all(["vision" not in key for key in model_config.keys()]):
        is_mm = False
    else:
        is_mm = True
    
    if is_mm:
        return getattr(importlib.import_module("flashrag.generator"), "HFMultiModalGenerator")(config, **params)
    else:
        if config["framework"] == "vllm":
            return getattr(importlib.import_module("flashrag.generator"), "VLLMGenerator")(config, **params)
        elif config["framework"] == "fschat":
            return getattr(importlib.import_module("flashrag.generator"), "FastChatGenerator")(config, **params)
        elif config["framework"] == "hf":
            if "t5" in arch.lower() or "bart" in arch.lower():
                return getattr(importlib.import_module("flashrag.generator"), "EncoderDecoderGenerator")(config, **params)
            else:
                return getattr(importlib.import_module("flashrag.generator"), "HFCausalLMGenerator")(config, **params)
        else:
            raise NotImplementedError


def get_retriever(config):
    r"""Automatically select retriever class based on config's retrieval method

    Args:
        config (dict): configuration with 'retrieval_method' key

    Returns:
        Retriever: retriever instance
    """
    if config["use_multi_retriever"]:
        # must load special class for manage multi retriever
        return getattr(importlib.import_module("flashrag.retriever"), "MultiRetrieverRouter")(config)

    if config["retrieval_method"] == "bm25":
        return getattr(importlib.import_module("flashrag.retriever"), "BM25Retriever")(config)
    else:
        try:
            model_config = AutoConfig.from_pretrained(config["retrieval_model_path"])
            arch = model_config.architectures[0]
            if "clip" in arch.lower():
                return getattr(importlib.import_module("flashrag.retriever"), "MultiModalRetriever")(config)
            else:
                return getattr(importlib.import_module("flashrag.retriever"), "DenseRetriever")(config)
        except:
            return getattr(importlib.import_module("flashrag.retriever"), "DenseRetriever")(config)


def get_reranker(config):
    model_path = config["rerank_model_path"]
    # get model config
    model_config = AutoConfig.from_pretrained(model_path)
    arch = model_config.architectures[0]
    if "forsequenceclassification" in arch.lower():
        return getattr(importlib.import_module("flashrag.retriever"), "CrossReranker")(config)
    else:
        return getattr(importlib.import_module("flashrag.retriever"), "BiReranker")(config)


def get_judger(config):
    judger_name = config["judger_name"]
    if "skr" in judger_name.lower():
        return getattr(importlib.import_module("flashrag.judger"), "SKRJudger")(config)
    elif "adaptive" in judger_name.lower():
        return getattr(importlib.import_module("flashrag.judger"), "AdaptiveJudger")(config)
    else:
        assert False, "No implementation!"


def get_refiner(config, retriever=None, generator=None):
    # 预定义默认路径字典
    DEFAULT_PATH_DICT = {
        "recomp_abstractive_nq": "fangyuan/nq_abstractive_compressor",
        "recomp:abstractive_tqa": "fangyuan/tqa_abstractive_compressor",
        "recomp:abstractive_hotpotqa": "fangyuan/hotpotqa_abstractive",
    }
    REFINER_MODULE = importlib.import_module("flashrag.refiner")

    refiner_name = config["refiner_name"]
    refiner_path = (
        config["refiner_model_path"]
        if config["refiner_model_path"] is not None
        else DEFAULT_PATH_DICT.get(refiner_name, None)
    )

    try:
        model_config = AutoConfig.from_pretrained(refiner_path)
        arch = model_config.architectures[0].lower()
        print(arch)
    except Exception as e:
        print("Warning", e)
        model_config, arch = "", ""

    if "recomp" in refiner_name:
        if model_config.model_type == "t5":
            refiner_class = "AbstractiveRecompRefiner"
        else:
            refiner_class = "ExtractiveRefiner"
    elif 'bert' in arch:
        refiner_class = "ExtractiveRefiner"
    elif 'T5' in arch or 'Bart' in arch:
        refiner_class = "AbstractiveRecompRefiner"
    elif "lingua" in refiner_name:
        refiner_class = "LLMLinguaRefiner"
    elif "selective-context" in refiner_name or "sc" in refiner_name:
        refiner_class = "SelectiveContextRefiner"
    elif "kg-trace" in refiner_name:
        return getattr(REFINER_MODULE, "KGTraceRefiner")(config, retriever, generator)
    else:
        raise ValueError("No implementation!")

    return getattr(REFINER_MODULE, refiner_class)(config)


def hash_object(o) -> str:
    """Returns a character hash code of arbitrary Python objects."""
    import hashlib
    import io
    import dill
    import base58

    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()

def extract_between(text: str, start_tag: str, end_tag: str):
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None