"""
Reference:
    Fangyuan Xu et al. "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation"
    in ICLR 2024.
    Official repo: https://github.com/carriex/recomp
"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline


# ###### Specified parameters ######
refiner_name = "recomp-abstractive" # recomp-extractive
refiner_model_path = "fangyuan/tqa_abstractive_compressor"
refiner_max_input_length = 1024
refiner_max_output_length = 512
# parameters for extractive compress
refiner_topk = 5
refiner_pooling_method = 'mean'
refiner_encode_max_length = 256


config_dict = {
    'refiner_name': refiner_name,
    'refiner_model_path': refiner_model_path,
    'refiner_max_input_length': refiner_max_input_length,
    'refiner_max_output_length': refiner_max_output_length,
    'refiner_topk': 5,
    'refiner_pooling_method': refiner_pooling_method,
    'refiner_encode_max_length': refiner_encode_max_length
}


# preparation
config = Config('my_config.yaml',config_dict)
all_split = get_dataset(config)
test_data = all_split['test']

pipeline = SequentialPipeline(config)
result = pipeline.run(test_data)
print(result)

