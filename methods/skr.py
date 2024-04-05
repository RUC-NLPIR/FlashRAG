"""
Reference:
    Yile Wang et al. "Self-Knowledge Guided Retrieval Augmentation for Large Language Models"
    in EMNLP Findings 2023.
    Official repo: https://github.com/THUNLP-MT/SKR/

Note:
    `skr-knn` need training data in inference stage to determain whether to retrieve. training data should in
    `.json` format in following format:
    format: 
        [
            {
                "question": ... ,  // question
                "judgement": "ir_better" / "ir_worse" / "same",  // judgement result, can be obtained by comparing 
                ...
            },
            ...
        ]

"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import ConditionalPipeline


# ###### Specified parameters ######
judger_name = 'skr'
model_path = 'princeton-nlp/sup-simcse-bert-base-uncased'
training_data_path = './sample_data/skr_training.json'

config_dict = {
    'judger_name': judger_name,
    'judger_model_path': model_path,
    'judger_training_data_path': training_data_path
}


# preparation
config = Config('my_config.yaml',config_dict)
all_split = get_dataset(config)
test_data = all_split['test']

pipeline = ConditionalPipeline(config)
result = pipeline.run(test_data)
print(result)
