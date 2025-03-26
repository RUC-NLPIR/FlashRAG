import json
import os
import warnings
from typing import Dict, Any, Union, List, Dict
import numpy as np
import datasets
import re
import langid
from transformers import AutoTokenizer, AutoModel, AutoConfig

_has_printed_instruction = False  # trigger instruction print once

def convert_numpy(obj: Union[Dict, list, np.ndarray, np.generic]) -> Any:
    """Recursively convert numpy objects in nested dictionaries or lists to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalars to native Python scalars
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj  # Return the object as-is if it's neither a dict, list, nor numpy type


def judge_zh(input_str: str):
    assert isinstance(input_str, str), input_str
    if len(input_str) == 0:
        return False
    detect_result = langid.classify(input_str)
    if detect_result[0] == 'zh':
        return True
    else:
        return False
    #return bool(re.search(r'[\u4e00-\u9fff]', input_str))


def convert_numpy(obj: Union[Dict, list, np.ndarray, np.generic]) -> Any:
    """Recursively convert numpy objects in nested dictionaries or lists to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalars to native Python scalars
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj  # Return the object as-is if it's neither a dict, list, nor numpy type
    
def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if last_hidden_state is None and pooling_method in ['mean', 'cls']:
        warnings.warn('last_hidden_state is None, using pooler_output instead.')
        pooling_method = 'pooler'

    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


def set_default_instruction(model_name, is_query=True, is_zh=False):
    instruction = ""
    if "e5" in model_name.lower():
        if is_query:
            instruction = "query: "
        else:
            instruction = "passage: "

    if "bge" in model_name.lower():
        if is_query:
            if "zh" in model_name.lower() or is_zh:
                instruction = "为这个句子生成表示以用于检索相关文章："
            else:
                instruction = "Represent this sentence for searching relevant passages: "

    return instruction


def parse_query(model_name, query_list, instruction=None, is_query=True):
    """
    processing query for different encoders
    """
    global _has_printed_instruction

    if isinstance(query_list, str):
        query_list = [query_list]

    if instruction is not None:
        instruction = instruction.strip() + " "
    else:
        instruction = set_default_instruction(model_name, is_query=is_query, is_zh=judge_zh(query_list[0]))
    
    if not _has_printed_instruction:
        if instruction == "":
            warnings.warn('Instruction is not set')
        else:
            print(f"Use `{instruction}` as retreival instruction")
        _has_printed_instruction = True
        
    query_list = [instruction + query for query in query_list]

    return query_list


def load_corpus(corpus_path: str):
    if corpus_path.endswith(".jsonl"):
        corpus = datasets.load_dataset('json', data_files=corpus_path, split="train")
    elif corpus_path.endswith(".parquet"):
        corpus = datasets.load_dataset('parquet', data_files=corpus_path, split="train")
        corpus = corpus.cast_column('image', datasets.Image())
    else:
        raise NotImplementedError("Corpus format not supported!")
    if 'contents' not in corpus.features:
        try:
            print("No `contents` field found in corpus, using `text` instead.")
            corpus = corpus.map(lambda x: {"contents": x["text"]})
        except:
            warnings.warn("No `contents` & `text` field found in corpus.")
    return corpus

def read_jsonl(file_path):
    with open(file_path, "r") as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)

            yield new_item


def load_docs(corpus, doc_idxs: List[int]):
    results = [corpus[int(idx)] for idx in doc_idxs]

    return results


def parse_image(image):
    from PIL import Image

    if isinstance(image, str):
        if image.startswith("http"):
            import requests

            image = Image.open(requests.get(image, stream=True).raw)
        else:
            image = Image.open(image)
    return image


def judge_image(x):
    from PIL import Image
    if isinstance(x, str):
        if x.startswith("http"):
            return True
        if os.path.exists(x):
            return True
    elif isinstance(x, Image.Image):
        return True
    else:
        warnings.warn('image type not supported')
    return False
    