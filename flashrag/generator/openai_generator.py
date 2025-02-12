import os
from typing import List
from copy import deepcopy
import warnings
from tqdm import tqdm
import numpy as np
import threading
import asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI
import tiktoken

_background_loop = None

def get_background_loop():
    global _background_loop
    if _background_loop is None:
        _background_loop = asyncio.new_event_loop()
        t = threading.Thread(target=lambda: _background_loop.run_forever(), daemon=True)
        t.start()
    return _background_loop

class OpenaiGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self._config = config
        self.update_config()
        
        # load openai client
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
    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()
    
    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()

    def update_base_setting(self):
        self.model_name = self._config["generator_model"]
        self.batch_size = self._config["generator_batch_size"]
        self.generation_params = self._config["generation_params"]

        self.openai_setting = self._config["openai_setting"]
        if self.openai_setting["api_key"] is None:
            self.openai_setting["api_key"] = os.getenv("OPENAI_API_KEY")

    def update_additional_setting(self):
        pass
    
    async def _get_response(self, messages: List, **params):
        response = await self.client.chat.completions.create(
            model=self.model_name, messages=messages, **params
        )
        if not response.choices:
            raise ValueError("No choices returned from API.")
        return response.choices[0]

    async def _get_batch_response(self, input_list: List[List], batch_size, **params):
        tasks = [self._get_response(messages, **params) for messages in input_list]
        all_results = []
        for idx in tqdm(range(0, len(tasks), batch_size), desc="Generation process: "):
            batch_tasks = tasks[idx: idx + batch_size]
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)
        return all_results

    async def _generate_async(self, input_list: List, batch_size=None, return_scores=False, **params) -> List[str]:
        if isinstance(input_list, dict):
            input_list = [[input_list]]
        elif isinstance(input_list[0], dict):
            input_list = [input_list]

        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        generation_params.pop("do_sample", None)

        max_tokens = params.pop("max_tokens", None) or params.pop("max_new_tokens", None)
        if max_tokens is not None:
            generation_params["max_tokens"] = max_tokens
        else:
            generation_params["max_tokens"] = generation_params.get(
                "max_tokens", generation_params.pop("max_new_tokens", None)
            )
        generation_params.pop("max_new_tokens", None)

        if return_scores:
            generation_params["logprobs"] = True
            warnings.warn("Set logprobs to True to get generation scores.")


        results = await self._get_batch_response(input_list, batch_size, **generation_params)

        response_texts = []
        scores = []
        for res in results:
            response_texts.append(res.message.content)
            if return_scores:
                score = np.exp([item.logprob for item in res.logprobs.content])
                scores.append(score)
        return (response_texts, scores) if return_scores else response_texts

    # ----------------- 同步包装接口 -----------------
    def generate(self, input_list: List, batch_size=None, return_scores=False, **params) -> List[str]:
        loop = get_background_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._generate_async(input_list, batch_size=batch_size, return_scores=return_scores, **params),
            loop
        )
        return future.result()
