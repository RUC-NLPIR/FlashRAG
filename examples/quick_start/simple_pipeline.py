import argparse
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--retriever_path', type=str)
args = parser.parse_args()

config_dict = {
                'data_dir': 'dataset/',
                'index_path': 'indexes/e5_flat_sample.index',
                'corpus_path': 'indexes/sample_data.jsonl',
                'model2path': {'e5': args.retriever_path, 'llama2-7B-chat': args.model_path},
                'generator_model': 'llama2-7B-chat',
                'retrieval_method': 'e5',
                'metrics': ['em','f1','sub_em'],
                'retrieval_topk': 1,
                'save_intermediate_data': True
            }

config = Config(config_dict = config_dict)

all_split = get_dataset(config)
test_data = all_split['test']
prompt_templete = PromptTemplate(
    config,
    system_prompt = "Answer the question based on the given document. \
                    Only give me the answer and do not output any other words. \
                    \nThe following are given documents.\n\n{reference}",
    user_prompt = "Question: {question}\nAnswer:"
)
pipeline = SequentialPipeline(config, prompt_template=prompt_templete)

output_dataset = pipeline.run(test_data,do_eval=True)
print("---generation output---")
print(output_dataset.pred)