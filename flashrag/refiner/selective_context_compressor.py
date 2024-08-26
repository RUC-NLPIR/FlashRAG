# Implementation of Selective-Context, modified from official repo: https://github.com/liyucheng09/Selective_Context
# Licensed under The MIT License


import re
from typing import List, Tuple
import spacy
import numpy as np
import os
import sys

sys.path.append("..")
from dataclasses import dataclass
from nltk.tokenize import word_tokenize
import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer


@dataclass
class LexicalUnits:
    unit_type: str
    text: List[str]
    self_info: List[float] = None

    def __add__(self, other):
        assert self.unit_type == other.unit_type, "Cannot add two different unit types"
        return LexicalUnits(self.unit_type, self.text + other.text, self.self_info + other.self_info)

    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplementedError()

    def add_to_head(self, token, self_info):
        return LexicalUnits(self.unit_type, [token] + self.text, [self_info] + self.self_info)

    def add_to_tail(self, token, self_info):
        return LexicalUnits(self.unit_type, self.text + [token], self.self_info + [self_info])


class SelectiveContext:

    def __init__(self, model_type="gpt2", model_path="openai-community/gpt2", lang="en"):

        self.model_type = model_type
        self.model_path = model_path
        self.lang = lang

        # this means we calculate self-information sentence by sentence
        self.sent_level_self_info = True

        self._prepare_phrase_tokenizer()
        self.sent_tokenize_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
        self.phrase_mask_token = ""
        self.sent_mask_token = "<...some content omitted.>"

        self._prepare_model()

    def _prepare_phrase_tokenizer(self):
        # we use space to tokenize sentence into phrases
        # for English, we should use `spacy.load("en_core_web_sm").add_pipe('merge_noun_chunks')`
        # for Chinese, use `nlp = spacy.load('zh_core_web_sm')`` directly
        lang = self.lang
        if lang == "en":
            self.nlp = spacy.load("en_core_web_lg", disable=["ner"])
            self.nlp.add_pipe("merge_noun_chunks")
        elif lang == "zh":
            self.nlp = spacy.load("zh_core_web_sm", disable=["ner"])

    def _prepare_model(self):
        # Load tokenizer
        if self.lang == "zh":
            self.tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
        elif self.lang == "en":
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        else:
            raise NotImplementedError()

        if self.model_type == "gpt2":
            if self.lang == "zh":
                self.model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
            else:
                self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.model.to("cuda")
            self.model.eval()

            print("model loaded")

            self.max_token_length = self.model.config.n_positions
            self.get_self_information = self._get_self_info_via_gpt2

        elif self.model_type == "curie":
            global openai
            import openai

            self.max_token_length = 2048

            self.get_self_information = self._get_self_info_via_curie

    def get_self_information(self, text: str) -> Tuple[List[str], List[float]]:
        # it takes text as input, and return a list of words and a list of self-information scores
        raise NotImplementedError

    def _get_self_info_via_gpt2(self, text: str) -> Tuple[List[str], List[float]]:
        if self.lang == "en":
            text = f"<|endoftext|>{text}"
        elif self.lang == "zh":
            text = f"[CLS]{text}"
        with torch.inference_mode(mode=True):
            encoding = self.tokenizer(
                text, max_length=1024, truncation=True, add_special_tokens=False, return_tensors="pt"
            )
            encoding = encoding.to(self.model.device)
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            self_info = -torch.log(probs)

        input_ids = encoding["input_ids"]
        input_ids_expaned = input_ids[:, 1:].unsqueeze(-1)

        tokens = [self.tokenizer.decode(token_) for token_ in input_ids.squeeze().tolist()[1:]]
        return tokens, self_info[:, :-1].gather(-1, input_ids_expaned).squeeze(-1).squeeze(0).tolist()

    def _get_self_info_via_curie(self, text: str) -> Tuple[List[str], List[float]]:
        num_retry = 3
        openai.api_key = os.environ["OPENAI_API_KEY"]

        for _ in range(num_retry):
            try:
                r = openai.Completion.create(
                    model="curie",
                    prompt=f"<|endoftext|>{text}",
                    max_tokens=0,
                    temperature=0,
                    echo=True,
                    logprobs=0,
                )
                break
            except Exception as e:
                print(e)
                time.sleep(1)

        result = r["choices"][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        self_info = [-logprob for logprob in logprobs]
        return tokens, self_info

    def _lexical_unit(self, sents):

        if self.sent_level_self_info:
            sent_self_info = []
            all_noun_phrases = []
            all_noun_phrases_info = []
            all_tokens = []
            all_token_self_info = []

            for sent in sents:
                # print(sent)
                tokens, self_info = self.get_self_information(sent)
                sent_self_info.append(np.mean(self_info))

                all_tokens.extend(tokens)
                all_token_self_info.extend(self_info)

                noun_phrases, noun_phrases_info = self._calculate_lexical_unit(tokens, self_info)

                # We need to add a space before the first noun phrase for every sentence except the first one
                if len(all_noun_phrases) != 0:
                    noun_phrases[0] = f" {noun_phrases[0]}"
                all_noun_phrases.extend(noun_phrases)
                all_noun_phrases_info.extend(noun_phrases_info)

            return [
                LexicalUnits("sent", text=sents, self_info=sent_self_info),
                LexicalUnits("phrase", text=all_noun_phrases, self_info=all_noun_phrases_info),
                LexicalUnits("token", text=all_tokens, self_info=all_token_self_info),
            ]

    def _calculate_lexical_unit(self, tokens, self_info):
        def _unit_info(tokens, self_info, units):
            current_unit_idx = 0
            current_position = 0
            unit_self_info = [[] for _ in range(len(units))]

            for idx, (token, info) in enumerate(zip(tokens, self_info)):
                current_position += len(token)
                if current_position == len(units[current_unit_idx]):
                    unit_self_info[current_unit_idx].append(info)
                    current_position = current_position - len(units[current_unit_idx])
                    current_unit_idx += 1
                elif current_position > len(units[current_unit_idx]):
                    counter_ = 1
                    current_position = current_position - len(units[current_unit_idx])
                    current_unit_idx += 1
                    while current_position >= len(units[current_unit_idx]):
                        counter_ += 1
                        current_position = current_position - len(units[current_unit_idx])
                        current_unit_idx += 1
                        if current_unit_idx >= len(units):
                            break
                    partial_info = info / counter_
                    for _ in range(counter_):
                        unit_self_info[(current_unit_idx - 1) - _].append(partial_info)
                else:
                    if token == " ":
                        continue
                    unit_self_info[current_unit_idx].append(info)

            unit_self_info_ = [np.mean(info) for info in unit_self_info]
            return unit_self_info_

        def _noun_phrases(sent):
            noun_phrases = []
            doc = self.nlp(sent)
            for index, chunk in enumerate(doc):
                if index == 0:
                    noun_phrases.append(chunk.text)
                else:
                    noun_phrases.append(doc[index - 1].whitespace_ + chunk.text)
            return noun_phrases

        if self.sent_level_self_info:
            # in this case, the self_info is for each sentence
            # we only need to calculate the self_info for each phrase

            sent = "".join(tokens)
            # noun_phrases = [chunk.text for chunk in self.nlp(sent).noun_chunks]
            noun_phrases = _noun_phrases(sent)
            # noun_phrases[-1] = noun_phrases[-1] + ' '
            noun_phrases_info = _unit_info(tokens, self_info, noun_phrases)

            return noun_phrases, noun_phrases_info

    def beautify_context(self, context: str) -> str:
        context = re.sub(r"\s+", " ", context)
        return context

    def self_info_mask(self, sents: List[str], self_info: List[float], mask_level):
        # mask_level: mask sentences, phrases, or tokens
        sents_after_mask = []
        masked_sents = []

        self.ppl_threshold = np.nanpercentile(self_info, self.mask_ratio * 100)

        for sent, info in zip(sents, self_info):
            if info < self.ppl_threshold:
                masked_sents.append(sent)
                sents_after_mask.append(self.mask_a_sent(sent, mask_level))
            else:
                sents_after_mask.append(sent)
        masked_context = " ".join(sents_after_mask) if mask_level == "sent" else "".join(sents_after_mask)

        return masked_context, masked_sents

    def mask_a_sent(self, sent, level):
        if level == "phrase":
            return self.phrase_mask_token
        elif level == "sent":
            if self.keep_leading_word:
                leading_few_words = " ".join(word_tokenize(sent)[: self.num_lead_words]) + " "
            else:
                leading_few_words = ""
            return leading_few_words + self.mask_token
        elif level == "token":
            return ""

    def __call__(self, text: str, reduce_ratio: float = 0.5, reduce_level: str = "phrase") -> List[str]:
        context = self.beautify_context(text)

        self.mask_ratio = reduce_ratio

        sents = re.split(self.sent_tokenize_pattern, context)
        sents = [sent.strip() for sent in sents if sent.strip()]

        # You want the reduce happen at sentence level, phrase level, or token level?
        assert reduce_level in [
            "sent",
            "phrase",
            "token",
        ], f"reduce_level should be one of ['sent', 'phrase', 'token'], got {reduce_level}"
        sent_lus, phrase_lus, token_lus = self._lexical_unit(sents)
        # print(phrase_lus, '^^^^')
        lexical_level = {"sent": sent_lus, "phrase": phrase_lus, "token": token_lus}

        # context is the reduced context, masked_sents denotes what context has been filtered out
        context, masked_sents = self.self_info_mask(
            lexical_level[reduce_level].text, lexical_level[reduce_level].self_info, reduce_level
        )
        return context, masked_sents
