import argparse
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import NaiveRAG


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--retriever_path', type=str)
args = parser.parse_args() 

config_dict = {
                'index_path': 'indexes/wiki_sample.index',
                'corpus_database_path': 'indexes/wiki_sample.db',
                'retriever_model2path': {'e5': args.retriever_path},
                'generator_model2path': {'llama2-7B-chat': args.model_path}
            }

config = Config(config_dict = config_dict)
all_split = get_dataset(config)
test_data = all_split['test']
pipeline = NaiveRAG(config)

output_dataset, result = pipeline.run(test_data)
print("---evaluation result---")
print(result)
print("---generation output---")
print(output_dataset.pred[:2])