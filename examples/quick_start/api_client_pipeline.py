from flashrag.config import Config
from flashrag.utils import get_dataset


config = Config('api_client_pipeline.yaml')
all_split = get_dataset(config)
test_data = all_split['test']


from flashrag.pipeline.pipeline import SequentialPipeline
pipeline = SequentialPipeline(config)
print(pipeline.run(test_data))


