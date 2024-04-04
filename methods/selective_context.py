"""
Reference:
    Yucheng Li et al. "Compressing Context to Enhance Inference Efficiency of Large Language Models"
    in EMNLP 2023.
    Official repo: https://github.com/liyucheng09/Selective_Context
"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline


from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline


# ###### Specified parameters ######
refiner_name = "selective-context"  
refiner_model_path = "openai-community/gpt2"

config_dict = {
    'refiner_name': refiner_name,
    'refiner_model_path': refiner_model_path,
    'sc_config':{
        'reduce_ratio': 0.3
    }
}


# preparation
config = Config('my_config.yaml',config_dict)
all_split = get_dataset(config)
test_data = all_split['test']

pipeline = SequentialPipeline(config)
result = pipeline.run(test_data)
print(result)

