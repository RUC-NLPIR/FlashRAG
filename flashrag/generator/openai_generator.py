import os
from typing import List
from copy import deepcopy
import warnings
from tqdm import tqdm
import numpy as np

import asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI
import tiktoken


class OpenaiGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.batch_size = config["generator_batch_size"]
        self.generation_params = config["generation_params"]

        self.openai_setting = config["openai_setting"]
        if self.openai_setting["api_key"] is None:
            self.openai_setting["api_key"] = os.getenv("OPENAI_API_KEY")

        if "api_type" in self.openai_setting and self.openai_setting["api_type"] == "azure":
            del self.openai_setting["api_type"]
            self.client = AsyncAzureOpenAI(**self.openai_setting)
        else:
            self.client = AsyncOpenAI(**self.openai_setting)
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except Exception as e:
            print("Error: ", e)
            warnings.warn("This model is not supported by tiktoken. Use gpt-3.5-turbo instead.")
            self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

    async def get_response(self, input: List, **params):
        response = await self.client.chat.completions.create(model=self.model_name, messages=input, **params)
        return response.choices[0]

    async def get_batch_response(self, input_list: List[List], batch_size, **params):
        total_input = [self.get_response(input, **params) for input in input_list]
        all_result = []
        for idx in tqdm(range(0, len(input_list), batch_size), desc="Generation process: "):
            batch_input = total_input[idx : idx + batch_size]
            batch_result = await asyncio.gather(*batch_input)
            all_result.extend(batch_result)

        return all_result

    def generate(self, input_list: List[List], batch_size=None, return_scores=False, **params) -> List[str]:
        # deal with single input
        if len(input_list) == 1 and isinstance(input_list[0], dict):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        # deal with generation params
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            generation_params.pop("do_sample")

        max_tokens = params.pop("max_tokens", None) or params.pop("max_new_tokens", None)
        if max_tokens is not None:
            generation_params["max_tokens"] = max_tokens
        else:
            generation_params["max_tokens"] = generation_params.get(
                "max_tokens", generation_params.pop("max_new_tokens", None)
            )
        generation_params.pop("max_new_tokens", None)

        if return_scores:
            if generation_params.get("logprobs") is not None:
                generation_params["logprobs"] = True
                warnings.warn("Set logprobs to True to get generation scores.")
            else:
                generation_params["logprobs"] = True

        if generation_params.get("n") is not None:
            generation_params["n"] = 1
            warnings.warn("Set n to 1. It can minimize costs.")
        else:
            generation_params["n"] = 1

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))

        # parse result into response text and logprob
        scores = []
        response_text = []
        for res in result:
            response_text.append(res.message.content)
            if return_scores:
                score = np.exp(list(map(lambda x: x.logprob, res.logprobs.content)))
                scores.append(score)
        if return_scores:
            return response_text, scores
        else:
            return response_text
