"""
Reference:
    Zhihong Shao et al. "Enhancing Retrieval-Augmented Large Language Models with Iterative
                         Retrieval-Generation Synergy"
    in EMNLP Findings 2023.
"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import ITERRETGENPipeline

iter_num = 3
# preparation
config = Config('my_config.yaml')
all_split = get_dataset(config)
test_data = all_split['test']

pipeline = ITERRETGENPipeline(config, iter_num=iter_num)
result = pipeline.run(test_data)
print(result)