"""
Reference:
    Huiqiang Jiang et al. "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models"
    in EMNLP 2023
    Huiqiang Jiang et al. "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression"
    in ICLR MEFoMo 2024.
    Official repo: https://github.com/microsoft/LLMLingua
"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline


# ###### Specified parameters ######
refiner_name = "longllmlingua" # 
refiner_model_path = "meta-llama/Llama-2-7b-hf"

config_dict = {
    'refiner_name': refiner_name,
    'refiner_model_path': refiner_model_path,
    'llmlingua_config':{
        'ratio': 0.55,
        'condition_in_question': 'after_condition',
        'reorder_context': 'sort',
        'dynamic_context_compression_ratio': 0.3,
        'condition_compare': True,
        'context_budget': "+100",
        'rank_method': 'longllmlingua'
    }
}


# preparation
config = Config('my_config.yaml',config_dict)
all_split = get_dataset(config)
test_data = all_split['test']

pipeline = SequentialPipeline(config)
result = pipeline.run(test_data)
print(result)

