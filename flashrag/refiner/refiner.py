from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flashrag.retriever.encoder import Encoder
from tqdm import tqdm
import re
import torch
import numpy as np

class BaseRefiner:
    r"""Base object of Refiner method"""

    def __init__(self, config):
        self.config = config
        self.name = config["refiner_name"]
        self.model_path = config["refiner_model_path"]
        self.device = config["device"]
        self.input_prompt_flag = config["refiner_input_prompt_flag"] if "refiner_input_prompt_flag" in config else False

    def run(self, item) -> str:
        r"""Get refining result.

        Args:
            item: dataset item, contains question, retrieval result...

        Returns:
            str: refining result of this item
        """
        pass

    def batch_run(self, dataset, batch_size=None) -> List[str]:
        return [self.run(item) for item in dataset]


class LLMLinguaRefiner(BaseRefiner):
    """Implementation for (Long)LLMLingua."""

    def __init__(self, config):
        super().__init__(config)
        default_config = {
            'use_llmlingua2': False,
            "rate": 0.55,
            "condition_in_question": "after_condition",
            "reorder_context": "sort",
            "dynamic_context_compression_ratio": 0.3,
            "condition_compare": True,
            "context_budget": "+100",
            "rank_method": "longllmlingua",
        }
        if "llmlingua_config" in config and config["llmlingua_config"] is not None:
            self.compress_config = config["llmlingua_config"]
        else:
            self.compress_config = default_config

        from flashrag.refiner.llmlingua_compressor import PromptCompressor

        if 'use_llmlingua2' in self.compress_config:
            use_llmlingua2 = self.compress_config.pop('use_llmlingua2')
        else:
            use_llmlingua2 = False
        self.refiner = PromptCompressor(model_name=self.model_path, use_llmlingua2=use_llmlingua2)

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

    def batch_run(self, dataset):
        output = []
        for item in tqdm(dataset, desc="Refining process: "):
            question = item.question
            # TODO: suit more cases
            if self.input_prompt_flag:
                input_prompt = item.prompt
                prompt_split = input_prompt.split("\n\n")
                # need fixed format prompt: instr + demon(retrieval results) + question
                instruction, question = prompt_split[0], prompt_split[-1]
                demonstration = "\n".join(prompt_split[1:-1])
                item_output = self.refiner.compress_prompt(
                    [i for i in demonstration.split("\n") if i != ""],
                    instruction=instruction,
                    question=question,
                    **self.compress_config,
                )
            else:
                retrieval_result = item.retrieval_result
                docs = self.format_reference(retrieval_result).split("\n")
                docs = [i for i in docs if i != ""]
                item_output = self.refiner.compress_prompt(
                    docs, instruction="", question=question, **self.compress_config
                )
            output.append(item_output["compressed_prompt"])
        return output


class SelectiveContextRefiner(BaseRefiner):
    """Implementation for Selective Context"""

    def __init__(self, config):
        super().__init__(config)
        from flashrag.refiner.selective_context_compressor import SelectiveContext

        default_config = {"reduce_ratio": 0.5}

        self.refiner = SelectiveContext(model_type="gpt2", model_path=self.model_path, lang="en")
        if "sc_config" in config and config["sc_config"] is not None:
            self.compress_config = config["sc_config"]
        else:
            self.compress_config = default_config

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

    def batch_run(self, dataset):
        # only use text
        all_inputs = []
        for item in dataset:
            retrieval_result = item.retrieval_result
            all_inputs.append(self.format_reference(retrieval_result))

        output = []
        for text in tqdm(all_inputs, desc="Refining process: "):
            compress_text, _ = self.refiner(text, **self.compress_config)
            output.append(compress_text)
        return output


class ExtractiveRefiner(BaseRefiner):
    """Implementation for Extractive compressor.
    Using retrieval method to select sentences or other granularity data.
    """

    def __init__(self, config):
        super().__init__(config)
        # number of keeping sentences
        self.topk = config["refiner_topk"]
        self.pooling_method = config["refiner_pooling_method"]
        self.encode_max_length = config["refiner_encode_max_length"]
        self.mini_batch_size = config['refiner_mini_batch_size'] if 'refiner_mini_batch_size' in config else 256
        # load model
        self.encoder = Encoder(
            model_name=self.name, 
            model_path=self.model_path, 
            pooling_method=self.pooling_method, 
            max_length=self.encode_max_length, 
            use_fp16=True
        )

    def batch_run(self, dataset, batch_size=16):
        questions = dataset.question
        # only use text
        retrieval_results = dataset.retrieval_result
        retrieval_results = [
            ["\n".join(doc_item["contents"].split("\n")[1:]) for doc_item in item_result]
            for item_result in retrieval_results
        ]

        # split into sentences: [[sent1, sent2,...], [...]]
        sent_lists = [
            [i.strip() for i in re.split(r"(?<=[.!?])\s+", " ".join(res)) if len(i.strip()) > 5]
            for res in retrieval_results
        ]
        score_lists = []  # matching scores, size == sent_lists
        for idx in tqdm(range(0, len(questions), batch_size), desc="Refining process: "):
            batch_questions = questions[idx : idx + batch_size]
            batch_sents = sent_lists[idx : idx + batch_size]
            question_embs = self.encoder.encode(batch_questions, is_query=True)
            
            flatten_batch_sents = sum(batch_sents, [])
            sent_embs = []
            for s_index in tqdm(range(0, len(flatten_batch_sents), self.mini_batch_size), desc='Sentence encoding..,'):
                mini_batch_sents = flatten_batch_sents[s_index:s_index+self.mini_batch_size]
                mini_sent_embs = self.encoder.encode(mini_batch_sents, is_query=False)
                sent_embs.append(mini_sent_embs)
            sent_embs = np.concatenate(sent_embs, axis=0)

            scores = question_embs @ sent_embs.T
            start_idx = 0
            for row_score, single_list in zip(scores, batch_sents):
                row_score = row_score.tolist()
                score_lists.append(row_score[start_idx : start_idx + len(single_list)])
                start_idx += len(single_list)

        # select topk sents
        retain_lists = []
        for sent_scores, sent_list in zip(score_lists, sent_lists):
            assert len(sent_scores) == len(sent_list)
            if len(sent_scores) < self.topk:
                retain_lists.append(sent_list)
                continue

            topk_idxs = torch.topk(torch.Tensor(sent_scores), min(self.topk, len(sent_scores))).indices.tolist()
            retain_lists.append([sent_list[idx] for idx in sorted(topk_idxs) if idx < len(sent_list)])

        return [" ".join(sents) for sents in retain_lists]


class AbstractiveRecompRefiner(BaseRefiner):
    """Implementation for Abstractive RECOMP compressor:
    RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation.
    """

    def __init__(self, config):
        super().__init__(config)

        self.max_input_length = config["refiner_max_input_length"]
        self.max_output_length = config["refiner_max_output_length"]

        # load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.cuda()
        self.model.eval()

    def batch_run(self, dataset, batch_size=2):
        # only use text
        retrieval_results = dataset.retrieval_result
        retrieval_results = [
            ["\n".join(doc_item["contents"].split("\n")[1:]) for doc_item in item_result]
            for item_result in retrieval_results
        ]

        # input processing in recomp training format
        format_inputs = [
            "Question: {question}\n Document: {document}\n Summary: ".format(
                question=item.question, document="\n".join(docs)
            )
            for item, docs in zip(dataset, retrieval_results)
        ]

        results = []
        for idx in tqdm(range(0, len(format_inputs), batch_size), desc="Refining process: "):
            batch_inputs = format_inputs[idx : idx + batch_size]
            batch_inputs = self.tokenizer(
                batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_length
            ).to(self.device)

            batch_outputs = self.model.generate(**batch_inputs, max_length=self.max_output_length)

            batch_outputs = self.tokenizer.batch_decode(
                batch_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            results.extend(batch_outputs)

        return results
