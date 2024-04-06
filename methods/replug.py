"""
Reference:
    Weijia Shi et al. "REPLUG: Retrieval-Augmented Black-Box Language Models".
"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import REPLUGPipeline

# preparation
config = Config('my_config.yaml')
all_split = get_dataset(config)
test_data = all_split['test']

pipeline = REPLUGPipeline(config)
result = pipeline.run(test_data)
print(result)
