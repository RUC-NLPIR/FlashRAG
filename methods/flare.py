"""
Reference:
    Zhengbao Jiang et al. "Active Retrieval Augmented Generation"
    in EMNLP 2023.
    Official repo: https://github.com/bbuing9/ICLR24_SuRe

"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import FLAREPipeline

# preparation
config = Config('my_config.yaml')
all_split = get_dataset(config)
test_data = all_split['test']

pipeline = FLAREPipeline(config)
result = pipeline.run(test_data)
print(result)
