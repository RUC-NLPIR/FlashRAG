import warnings
import os
import importlib
from flashrag.dataset.dataset import Dataset

def get_dataset(config):
    dataset_path = config['dataset_path']
    all_split = config['split']

    split_dict = {
                'train': None,
                'dev': None,
                'test': None
            }
    for split in all_split:
        split_path = os.path.join(dataset_path, f'{split}.jsonl')
        if not os.path.exists(split_path):
            print(f"{split} file not exists!")
            continue
        if split == "test":
            split_dict[split] = Dataset(config, 
                                        split_path, 
                                        sample_num = config['test_sample_num'], 
                                        random_sample = config['random_sample'])
        else:
            split_dict[split] = Dataset(config, split_path)
     
    return split_dict

def get_generator(config):
    r"""Automatically select generator class based on config."""
    if "t5" in config['generator_model'] or "bart" in config['generator_model']:
        return getattr(
            importlib.import_module("flashrag.generator"), 
            "EncoderDecoderGenerator"
        )(config)
    else:
        return getattr(
                importlib.import_module("flashrag.generator"), 
                "CausalLMGenerator"
            )(config)

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
