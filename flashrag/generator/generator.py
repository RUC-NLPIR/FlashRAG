from typing import List
from copy import deepcopy
from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    AutoConfig,
)


class BaseGenerator:
    """`BaseGenerator` is a base object of Generator model."""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.model_path = config["generator_model_path"]

        self.max_input_len = config["generator_max_input_len"]
        self.batch_size = config["generator_batch_size"]
        self.device = config["device"]
        self.gpu_num = torch.cuda.device_count()

        self.generation_params = config["generation_params"]

    def generate(self, input_list: list) -> List[str]:
        """Get responses from the generater.

        Args:
            input_list: it contains input texts, each item represents a sample.

        Returns:
            list: contains generator's response of each input sample.
        """
        pass


class EncoderDecoderGenerator(BaseGenerator):
    """Class for encoder-decoder model"""

    def __init__(self, config):
        super().__init__(config)
        self.fid = config["use_fid"]
        model_config = AutoConfig.from_pretrained(self.model_path)
        arch = model_config.architectures[0].lower()
        if "t5" in arch:
            if self.fid:
                from flashrag.generator.fid import FiDT5

                self.model = FiDT5.from_pretrained(self.model_path)
            else:
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_path
                )
        else:
            if self.fid:
                assert False, "FiD only support T5"
            self.model = BartForConditionalGeneration.from_pretrained(
                self.model_path
            )
        self.model.cuda()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def encode_passages(self, batch_text_passages: List[List[str]]):
        passage_ids, passage_masks = [], []
        for k, text_passages in enumerate(batch_text_passages):
            p = self.tokenizer.batch_encode_plus(
                text_passages,
                max_length=self.max_input_len,
                pad_to_max_length=True,
                return_tensors="pt",
                truncation=True,
            )
            passage_ids.append(p["input_ids"][None])
            passage_masks.append(p["attention_mask"][None])

        passage_ids = torch.cat(passage_ids, dim=0)
        passage_masks = torch.cat(passage_masks, dim=0)
        return passage_ids, passage_masks.bool()

    @torch.inference_mode(mode=True)
    def generate(self, input_list: List, batch_size=None, **params):
        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # deal stop params
        stop_sym = None
        if "stop" in generation_params:
            from flashrag.generator.stop_word_criteria import StopWordCriteria

            stop_sym = generation_params.pop("stop")
            stopping_criteria = [
                StopWordCriteria(
                    tokenizer=self.tokenizer,
                    prompts=input_list,
                    stop_words=stop_sym,
                )
            ]
            generation_params["stopping_criteria"] = stopping_criteria

        max_tokens = params.pop("max_tokens", None) or params.pop(
            "max_new_tokens", None
        )
        if max_tokens is not None:
            generation_params["max_new_tokens"] = max_tokens
        else:
            generation_params["max_new_tokens"] = generation_params.get(
                "max_new_tokens", generation_params.pop("max_tokens", None)
            )
        generation_params.pop("max_tokens", None)

        responses = []
        for idx in trange(
            0, len(input_list), batch_size, desc="Generation process: "
        ):
            batched_prompts = input_list[idx : idx + batch_size]
            if self.fid:
                # assume each input in input_list is a list, contains K string
                input_ids, attention_mask = self.encode_passages(
                    batched_prompts
                )
                inputs = {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                }
            else:
                inputs = self.tokenizer(
                    batched_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_len,
                ).to(self.device)

            # TODO: multi-gpu inference
            outputs = self.model.generate(**inputs, **generation_params)

            outputs = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            responses += outputs

        return responses


class VLLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on vllm."""

    def __init__(self, config):
        super().__init__(config)

        from vllm import LLM

        if "gpu_memory_utilization" not in config:
            gpu_memory_utilization = 0.85
        else:
            gpu_memory_utilization = config["gpu_memory_utilization"]
        if self.gpu_num != 1 and self.gpu_num % 2 != 0:
            tensor_parallel_size = self.gpu_num - 1
        else:
            tensor_parallel_size = self.gpu_num

        self.lora_path = (
            None
            if "generator_lora_path" not in config
            else config["generator_lora_path"]
        )
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True
        if self.use_lora:
            self.model = LLM(
                self.model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_lora=True,
                max_lora_rank=64,
                max_logprobs=32016,
            )
        else:
            self.model = LLM(
                self.model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_logprobs=32016,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    @torch.inference_mode(mode=True)
    def generate(
        self,
        input_list: List[str],
        return_raw_output=False,
        return_scores=False,
        **params,
    ):
        from vllm import SamplingParams

        if isinstance(input_list, str):
            input_list = [input_list]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            generation_params.pop("do_sample")

        max_tokens = params.pop("max_tokens", None) or params.pop(
            "max_new_tokens", None
        )
        if max_tokens is not None:
            generation_params["max_tokens"] = max_tokens
        else:
            generation_params["max_tokens"] = generation_params.get(
                "max_tokens", generation_params.pop("max_new_tokens", None)
            )
        generation_params.pop("max_new_tokens", None)

        # fix for llama3
        if "stop" in generation_params:
            generation_params["stop"].append("<|eot_id|>")
        else:
            generation_params["stop"] = ["<|eot_id|>"]

        if return_scores:
            if "logprobs" not in generation_params:
                generation_params["logprobs"] = 100

        sampling_params = SamplingParams(**generation_params)

        if self.use_lora:
            from vllm.lora.request import LoRARequest

            outputs = self.model.generate(
                input_list,
                sampling_params,
                lora_request=LoRARequest("lora_module", 1, self.lora_path),
            )
        else:
            outputs = self.model.generate(input_list, sampling_params)

        if return_raw_output:
            base_output = outputs
        else:
            generated_texts = [output.outputs[0].text for output in outputs]
            base_output = generated_texts
        if return_scores:
            scores = []
            for output in outputs:
                logprobs = output.outputs[0].logprobs
                scores.append(
                    [
                        np.exp(list(score_dict.values())[0].logprob)
                        for score_dict in logprobs
                    ]
                )
            return base_output, scores
        else:
            return base_output


class HFCausalLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on hf."""

    def __init__(self, config, model=None):
        super().__init__(config)
        self.config = config
        lora_path = (
            None
            if "generator_lora_path" not in config
            else config["generator_lora_path"]
        )
        self.model, self.tokenizer = self._load_model(model=model)
        self.use_lora = False
        if lora_path is not None:
            self.use_lora = True
            self.model.load_adapter(lora_path)

    def _load_model(self, model=None):
        r"""Load model and tokenizer for generator."""
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model.cuda()
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if "qwen" not in self.model_name:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer

    def add_new_tokens(
        self, token_embedding_path, token_name_func=lambda idx: f"[ref{idx+1}]"
    ):
        del self.model
        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
        )
        # get original embedding weight matrix
        embedding_layer = self.model.get_input_embeddings()
        embedding_weights = embedding_layer.weight
        original_vocab_size, embedding_dim = embedding_weights.shape
    
        new_tokens_weights = torch.load(token_embedding_path)
        new_tokens_length = new_tokens_weights.shape[0]

        # expand vocabulary
        new_tokens = [token_name_func(idx) for idx in range(new_tokens_length)]
        self.tokenizer.add_tokens(new_tokens)

        # create new embedding matrix
        new_vocab_size = original_vocab_size + new_tokens_length
        new_embedding_weights = torch.zeros(new_vocab_size, embedding_dim)

        # copy original embeddings to the new weights
        new_embedding_weights[:original_vocab_size, :] = embedding_weights

        # append virtual token embeddings to the new weights
        for token, embedding in zip(new_tokens, new_tokens_weights):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            new_embedding_weights[token_id] = embedding

        # update the embedding table
        # note: we should avoid using the function resize_token_embeddings() because this function will also change the lm_head of the model
        embedding_layer.weight.data = new_embedding_weights
        self.model.eval()
        self.model.cuda()

    @torch.inference_mode(mode=True)
    def generate(
        self,
        input_list: List[str],
        batch_size=None,
        return_scores=False,
        return_dict=False,
        **params,
    ):
        """Generate batches one by one. The generated content needs to exclude input."""

        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # deal stop params
        stop_sym = None
        if "stop" in generation_params:
            from flashrag.generator.stop_word_criteria import StopWordCriteria

            stop_sym = generation_params.pop("stop")
            stopping_criteria = [
                StopWordCriteria(
                    tokenizer=self.tokenizer,
                    prompts=input_list,
                    stop_words=stop_sym,
                )
            ]
            generation_params["stopping_criteria"] = stopping_criteria

        max_tokens = params.pop("max_tokens", None) or params.pop(
            "max_new_tokens", None
        )
        if max_tokens is not None:
            generation_params["max_new_tokens"] = max_tokens
        else:
            generation_params["max_new_tokens"] = generation_params.get(
                "max_new_tokens", generation_params.pop("max_tokens", None)
            )
        generation_params.pop("max_tokens", None)

        # set eos token for llama
        if "llama" in self.model_name.lower():
            extra_eos_tokens = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            if "eos_token_id" in generation_params:
                generation_params["eos_token_id"].extend(extra_eos_tokens)
            else:
                generation_params["eos_token_id"] = extra_eos_tokens

        responses = []
        scores = []
        generated_token_ids = []
        generated_token_logits = []

        for idx in trange(
            0, len(input_list), batch_size, desc="Generation process: "
        ):
            torch.cuda.empty_cache()
            batched_prompts = input_list[idx : idx + batch_size]
            inputs = self.tokenizer(
                batched_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_len,
            ).to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                **generation_params,
            )

            generated_ids = outputs.sequences
            logits = torch.stack(outputs.scores, dim=1).softmax(-1)
            generated_ids = generated_ids[:, inputs["input_ids"].shape[-1] :]
            gen_score = (
                torch.gather(logits, 2, generated_ids[:, :, None])
                .squeeze(-1)
                .cpu()
                .tolist()
            )
            scores.extend(gen_score)

            # get additinoal info
            if return_dict:
                batch_generated_token_ids = generated_ids.detach().cpu()
                batch_generated_token_logits = (
                    torch.cat(
                        [
                            token_scores.unsqueeze(1)
                            for token_scores in outputs.scores
                        ],
                        dim=1,
                    )
                    .detach()
                    .cpu()
                )
                if (
                    batch_generated_token_ids.shape[1]
                    < generation_params["max_new_tokens"]
                ):
                    real_batch_size, num_generated_tokens = (
                        batch_generated_token_ids.shape
                    )
                    padding_length = (
                        generation_params["max_new_tokens"]
                        - num_generated_tokens
                    )
                    padding_token_ids = torch.zeros(
                        (real_batch_size, padding_length),
                        dtype=batch_generated_token_ids.dtype,
                    ).fill_(self.tokenizer.pad_token_id)
                    padding_token_logits = torch.zeros(
                        (
                            real_batch_size,
                            padding_length,
                            batch_generated_token_logits.shape[-1],
                        ),
                        dtype=batch_generated_token_logits.dtype,
                    )
                    batch_generated_token_ids = torch.cat(
                        [batch_generated_token_ids, padding_token_ids], dim=1
                    )
                    batch_generated_token_logits = torch.cat(
                        [batch_generated_token_logits, padding_token_logits],
                        dim=1,
                    )
                generated_token_ids.append(batch_generated_token_ids)
                generated_token_logits.append(batch_generated_token_logits)

            for i, generated_sequence in enumerate(outputs.sequences):
                input_ids = inputs["input_ids"][i]
                text = self.tokenizer.decode(
                    generated_sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    )
                new_text = text[prompt_length:]

                if stop_sym is not None:
                    strip_stopword = True
                    # Find the first occurrence of any stop word
                    lower_stop_index = len(new_text)  # Default to end of text
                    for sym in stop_sym:
                        stop_index = new_text.find(sym)
                        if stop_index != -1:
                            # Adjust stop index based on whether we're stripping the stop word
                            stop_index += 0 if strip_stopword else len(sym)
                            lower_stop_index = min(stop_index, lower_stop_index)

                    # Cut the text at the first stop word found (if any)
                    new_text = new_text[:lower_stop_index]

                responses.append(new_text.strip())

        if return_dict:
            generated_token_ids = torch.cat(generated_token_ids, dim=0)
            generated_token_logits = torch.cat(generated_token_logits, dim=0)
            return {
                "generated_token_ids": generated_token_ids,
                "generated_token_logits": generated_token_logits,
                "responses": responses,
                "scores": scores,
            }

        if return_scores:
            return responses, scores
        else:
            return responses

    @torch.inference_mode(mode=True)
    def cal_gen_probs(self, prev, next):
        input_ids = self.tokenizer.encode(prev, add_special_tokens=False)
        target_ids = self.tokenizer.encode(next, add_special_tokens=False)
        context_ids = input_ids + target_ids
        context_tensor = torch.tensor([context_ids]).to(self.device)
        with torch.no_grad():
            outputs = self.model(context_tensor)
            logits = outputs.logits
            logits = logits[0, len(input_ids) - 1 : len(context_ids) - 1, :]
            logits = logits.to(torch.float32).detach().cpu()
            # softmax to normalize
            probs = torch.softmax(logits, dim=-1)
            # obtain probs of target_ids
            target_probs = probs[range(len(target_ids)), target_ids].numpy()

        return logits, target_probs


class FastChatGenerator(HFCausalLMGenerator):
    def __init__(self, config, model=None):
        super().__init__(config)

    def _load_model(self, model=None):
        r"""Load model and tokenizer for generator."""

        def get_gpu_memory(max_gpus=None):
            """Get available memory for each GPU."""
            gpu_memory = []
            num_gpus = (
                torch.cuda.device_count()
                if max_gpus is None
                else min(max_gpus, torch.cuda.device_count())
            )
            for gpu_id in range(num_gpus):
                with torch.cuda.device(gpu_id):
                    device = torch.cuda.current_device()
                    gpu_properties = torch.cuda.get_device_properties(device)
                    total_memory = gpu_properties.total_memory / (1024**3)
                    allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                    available_memory = total_memory - allocated_memory
                    gpu_memory.append(available_memory)
            return gpu_memory

        if model is None:
            from fastchat.model import load_model

            if "gpu_memory_utilization" not in self.config:
                gpu_memory_utilization = 0.85
            else:
                gpu_memory_utilization = self.config["gpu_memory_utilization"]
            max_gpu_memory = None
            if self.gpu_num != 1:
                available_gpu_memory = get_gpu_memory(self.gpu_num)
                max_gpu_memory = (
                    str(int(min(available_gpu_memory) * gpu_memory_utilization))
                    + "GiB"
                )

            model, tokenizer = load_model(
                self.model_path,
                device="cuda",
                num_gpus=self.gpu_num,
                max_gpu_memory=max_gpu_memory,
                load_8bit=False,
                cpu_offloading=False,
                debug=False,
            )

        else:
            model.cuda()
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if "qwen" not in self.model_name:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer
