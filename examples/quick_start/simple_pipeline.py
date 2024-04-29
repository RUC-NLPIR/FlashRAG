import argparse
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--retriever_path', type=str)
args = parser.parse_args() 

config_dict = { 
                'data_dir': 'dataset/',
                'index_path': 'indexes/e5_flat_sample.index',
                'corpus_path': 'indexes/sample_data.jsonl',
                'retriever_model2path': {'e5': args.retriever_path},
                'generator_model2path': {'llama2-7B-chat': args.model_path},
                'metrics': ['em','f1','sub_em']
            }

config = Config(config_dict = config_dict)

all_split = get_dataset(config)
test_data = all_split['test']
pipeline = SequentialPipeline(config)

output_dataset = pipeline.run(test_data,do_eval=True)
print("---generation output---")
print(output_dataset.pred)