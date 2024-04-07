"""
Reference:
    Jaehyung Kim et al. "SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs"
    in ICLR 2024
    Official repo: https://github.com/bbuing9/ICLR24_SuRe

"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SuRePipeline

# preparation
config = Config('my_config.yaml')
all_split = get_dataset(config)
test_data = all_split['test']

pipeline = SuRePipeline(config)
result = pipeline.run(test_data)
print(result)
