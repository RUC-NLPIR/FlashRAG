"""
Reference:
    Yile Wang et al. "Self-Knowledge Guided Retrieval Augmentation for Large Language Models"
    in EMNLP Findings 2023.
    Official repo: https://github.com/THUNLP-MT/SKR/
"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import ConditionalPipeline


# ###### Specified parameters ######
judger_name = 'skr'
model_path = 'princeton-nlp/sup-simcse-bert-base-uncased'

config_dict = {
    'judger_name': judger_name,
    'judger_model_path': model_path
}


# preparation
config = Config('my_config.yaml',config_dict)
all_split = get_dataset(config)
test_data = all_split['test']

pipeline = ConditionalPipeline(config)
result = pipeline.run(test_data)
print(result)
