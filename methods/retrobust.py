"""
Reference:
    Ori Yoran et al. "Making Retrieval-Augmented Language Models Robust to Irrelevant Context"
    in ICLR 2024.
    Official repo: https://github.com/oriyor/ret-robust
"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.utils import retrobust_pred_parse


# preparation
config = Config('my_config.yaml')
all_split = get_dataset(config)
test_data = all_split['test']

pipeline = SequentialPipeline(config)
# use specify prediction parse function 
result = pipeline.run(test_data, pred_process_fun=retrobust_pred_parse)
print(result)