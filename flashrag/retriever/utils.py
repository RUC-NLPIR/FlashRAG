import json
import warnings
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig


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
            if 'zh' in model_name.lower() or is_zh:
                instruction = "为这个句子生成表示以用于检索相关文章："
            else:
                instruction = "Represent this sentence for searching relevant passages: "

    return instruction


def parse_query(model_name, query_list, instruction=None):
    """
    processing query for different encoders
    """

    def is_zh(str):
        import unicodedata

        zh_char = 0
        for c in str:
            try:
                if "CJK" in unicodedata.name(c):
                    zh_char += 1
            except:
                continue
        if zh_char / len(str) > 0.2:
            return True
        else:
            return False

    if isinstance(query_list, str):
        query_list = [query_list]
        
    if instruction is not None:
        instruction = instruction.strip() + " "
    else:
        instruction = set_default_instruction(model_name, is_query=True, is_zh=is_zh(query_list[0]))
    print(f"Use `{instruction}` as retreival instruction")

    query_list = [instruction + query for query in query_list]

    return query_list

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train")
    return corpus


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)

            yield new_item


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]

    return results
