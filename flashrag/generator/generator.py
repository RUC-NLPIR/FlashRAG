import torch
import torch.nn.functional as F
import torch.nn as nn
import collections
from typing import List, Dict
import re
import openai
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, T5ForConditionalGeneration, BartForConditionalGeneration



class BaseGenerator:
    r"""`BaseGenerator` is a base object of Generator model.
    
    """

    def __init__(self, config):
        self.model_name = config['generator_model']
        self.model_path = config['generator_model_path']

        self.max_input_len = config['generator_max_input_len']
        self.batch_size = config['generator_batch_size']
        self.device = config['device']
        self.gpu_num = torch.cuda.device_count()

        self.generation_params = config['generation_params']
    
    def generate(self, input_list: list) -> List[str]:
        r"""Get responses from the generater.

        Args:
            input_list: it contains input texts, each item represents a sample.
        
        Returns:
            list: contains generator's response of each input sample.
        """
        pass


class EncoderDecoderGenerator(BaseGenerator):
    r"""Class for encoder-decoder model"""
    def __init__(self, config):
        super().__init__(config)
        if "t5" in self.model_name: 
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        else:
            self.model = BartForConditionalGeneration.from_pretrained(self.model_path)
        self.model.cuda()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)


    @torch.no_grad()
    def generate(self, input_list: List, batch_size=None, **params):
        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size
        
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # deal stop params
        if 'stop' in generation_params:
            from transformers.generation.stopping_criteria import StoppingCriteriaList,StopAtSpecificTokenCriteria
            stop_sym = generation_params.pop('stop')
            stopping_criteria = StoppingCriteriaList()
            for sym in stop_sym:
                token_id = self.tokenizer.encode(sym)[0]
                stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[token_id]))
            generation_params['stopping_criteria'] = stopping_criteria

        if 'max_tokens' in generation_params:
            if 'max_tokens' in params:
                generation_params['max_new_tokens'] = params.pop('max_tokens')
            else:
                generation_params['max_new_tokens'] = generation_params.pop('max_tokens')

        responses = []
        for idx in tqdm(range(0, len(input_list), batch_size), desc='Generation process: '):
            batched_prompts = input_list[idx:idx+batch_size]
            inputs = self.tokenizer(batched_prompts, 
                                    return_tensors="pt", 
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_input_len
                                ).to(self.device)
            
            # TODO: multi-gpu inference
            outputs = self.model.generate(
                **inputs,
                **generation_params
            )

            outputs = self.tokenizer.batch_decode(outputs, 
                                                  skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=False)

            responses += outputs

        return responses


class VLLMGenerator(BaseGenerator):
    r"""Class for decoder-only generator, based on vllm. 
    """
    def __init__(self, config):
        super().__init__(config)
        
        from vllm import LLM
        if 'vllm_gpu_memory_utilization' not in config:
            gpu_memory_utilization = 0.9
        else:
            gpu_memory_utilization = config['vllm_gpu_memory_utilization']
        if self.gpu_num != 1 and self.gpu_num%2 == 0:
            tensor_parallel_size = self.gpu_num - 1
        else:
            tensor_parallel_size = self.gpu_num
        self.model = LLM(self.model_path, 
                         tensor_parallel_size = tensor_parallel_size,
                         gpu_memory_utilization = gpu_memory_utilization
                        )

        self.lora_path = None if 'generator_lora_path' not in config else config['generator_lora_path']
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True

            
    @torch.no_grad()
    def generate(self, input_list, return_raw_output=False, return_scores=False, **params):
        from vllm import SamplingParams
        if isinstance(input_list, str):
            input_list = [input_list]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if 'max_new_tokens' in generation_params:
            if 'max_new_tokens' in params:
                generation_params['max_tokens'] = params.pop('max_new_tokens')
            else:
                generation_params['max_tokens'] = generation_params.pop('max_new_tokens')
        if return_scores:
            if 'logprobs' not in generation_params:
                generation_params['logprobs'] = 100
        sampling_params = SamplingParams(**generation_params)
    
        if self.use_lora:
            from vllm.lora.request import LoRARequest
            outputs = self.model.generate(
                input_list,
                sampling_params,
                lora_request=LoRARequest('lora_module', 1, self.lora_path)
            )
        else:
            outputs = self.model.generate(
                input_list,
                sampling_params
            )

        if return_raw_output:
            base_output =  outputs
        else:
            generated_texts = [output.outputs[0].text for output in outputs]
            base_output = generated_texts
        if return_scores:
            scores = []
            for output in outputs:
                logprobs = output.outputs[0].logprobs
                scores.append(
                    [np.exp(list(score_dict.values())[0]) for score_dict in logprobs]
                )
            return base_output, scores
        else:
            return base_output


class CausalLMGenerator(BaseGenerator):
    r"""Class for decoder-only generator, based on hf. 
    """
    def __init__(self, config, model=None):
        super().__init__(config)
        lora_path = None if 'generator_lora_path' not in config else config['generator_lora_path']
        self.model, self.tokenizer = self._load_model(model=model)
        self.use_lora = False
        if lora_path is not None:
            self.use_lora = True
            import peft
            self.model.load_adapter(lora_path)
    
    def _load_model(self, model=None):
        r"""Load model and tokenizer for generator.
        
        """
        if model is None:
            from fastchat.model import load_model
            model, tokenizer = load_model(self.model_path,
                                            device = 'cuda', 
                                            num_gpus = self.gpu_num,
                                            load_8bit = False,
                                            cpu_offloading = False,
                                            debug = False,)
            #model = AutoModelForCausalLM.from_pretrained(self.model_path)
            #tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = model.to(self.device)
            
        else:
            model.cuda()
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model.eval()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer
    
    @torch.no_grad()
    def generate(self, input_list, batch_size=None, return_scores=False, **params):
        r"""Generate batches one by one. The generated content needs to exclude input.
    
        """
        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # deal stop params
        if 'stop' in generation_params:
            from transformers.generation.stopping_criteria import StoppingCriteriaList,StopAtSpecificTokenCriteria
            stop_sym = generation_params.pop('stop')
            stopping_criteria = StoppingCriteriaList()
            for sym in stop_sym:
                token_id = self.tokenizer.encode(sym)[0]
                stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[token_id]))
            generation_params['stopping_criteria'] = stopping_criteria

        if 'max_tokens' in generation_params:
            if 'max_tokens' in params:
                generation_params['max_new_tokens'] = params.pop('max_tokens')
            else:
                generation_params['max_new_tokens'] = generation_params.pop('max_tokens')

        responses = []
        scores = []
        for idx in tqdm(range(0, len(input_list), batch_size), desc='Generation process: '):
            batched_prompts = input_list[idx:idx+batch_size]
            inputs = self.tokenizer(batched_prompts, 
                                    return_tensors="pt", 
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_input_len
                                ).to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                **generation_params
            )
            
            
            generated_ids = outputs.sequences
            logits = torch.stack(outputs.scores, dim=1).softmax(-1)
            generated_ids = generated_ids[:, inputs['input_ids'].shape[-1]:]
            gen_score = torch.gather(logits, 2, generated_ids[:, :, None]).squeeze(-1).cpu().tolist()
            scores.extend(gen_score)

            for i, generated_sequence in enumerate(outputs.sequences):
                input_ids = inputs['input_ids'][i]
                text = self.tokenizer.decode(
                            generated_sequence, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False
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
                responses.append(new_text.strip())
        if return_scores:
            return responses, scores
        else:
            return responses
    

    def cal_gen_probs(self, prev, next):
        input_ids = self.tokenizer.encode(prev, add_special_tokens=False)
        target_ids = self.tokenizer.encode(next, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids + target_ids]).to(self.device)
        target_tensor = torch.tensor([[-100] * len(input_ids) + target_ids]).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor, labels=target_tensor)
            logits = outputs.logits
            logits = logits[0, len(input_ids):, :]
            logits = logits.to(torch.float32).detach().cpu().numpy()
            logits = logits[range(len(target_ids)), target_ids]
        
        return target_ids, logits
