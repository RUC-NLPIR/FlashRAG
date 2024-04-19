import faiss
import json
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
from sqlite_utils import Database
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig, T5EncoderModel


def load_model(
        model_path: str, 
        use_fp16: bool = False
    ):
    model_config = AutoConfig.from_pretrained(model_path)
    model_class = AutoModel
    #model_class = T5EncoderModel if "t5" in model_config.architectures[0].lower() else AutoModel

    model = model_class.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    return model, tokenizer


def pooling(
        pooler_output,
        last_hidden_state,
        attention_mask = None,
        pooling_method = "mean"
    ):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

def base_content_function(item):
    if 'title' in item:
        return "\"{}\"\n{}".format(item['title'], item['text'])
    else:
        return item['text']

def load_database(database_path: str):
    db = Database(database_path)
    corpus = db['docs']
    return corpus
    

def read_jsonl(file_path, content_function=None):
    with open(file_path, "r") as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)
            if content_function:
                new_item['contents'] = content_function(new_item)
            
            yield new_item

def load_corpus(
        corpus_path: str,
        content_function: callable = lambda item: "\"{}\"\n{}".format(item['title'], item['text'])
    ):
    raw_corpus_sample = next(read_jsonl(corpus_path))
    have_contents = 'contents' in raw_corpus_sample

    import subprocess
    out = subprocess.getoutput("wc -l %s" % corpus_path)
    corpus_size = int(out.split()[0])
    
    if not have_contents:
        corpus = read_jsonl(corpus_path, content_function)
    else:
        # use original 'contents' key
        corpus = read_jsonl(corpus_path)

    return corpus, have_contents, corpus_size

def load_docs(corpus, doc_idxs, content_function=base_content_function):
    doc_ids =  [str(idx) for idx in doc_idxs]
    query = 'id IN ({})'.format(','.join('?' * len(doc_ids)))
    results = corpus.rows_where(query, doc_ids)
    results = list(results)

    # match the corresponding idx
    id2item = {item['id']:item for item in results}
    results = [id2item[id] for id in doc_ids]

    # add content field
    for item in results:
        if 'contents' not in item:
            item['contents'] = content_function(item)

    return results