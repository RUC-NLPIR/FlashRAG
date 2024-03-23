import json
import os
import argparse
from flashrag.config import Config
from flashrag.utils import get_retriever,get_dataset


parser = argparse.ArgumentParser(description = "Retrieving docs for dataset.")
# Basic parameters
parser.add_argument('--dataset', type=str)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()


dataset_name = args.dataset
save_dir = args.save_dir

config_dict = {"dataset_name": dataset_name}
config = Config('my_config.yaml', config_dict)

save_dir = os.path.join(save_dir, dataset_name, config['retrieval_method'])
os.makedirs(save_dir, exist_ok=True)

retriever = get_retriever(config)
all_split = get_dataset(config)

for split, data in all_split.items():
    if data is None:
        continue
    print(split)

    query_list = data.question
    results = retriever.batch_search(query_list, 20)

    # save format: {"id":.., "question":..., "retrieve_docs": [{doc1}, {doc2}]}

    output = [{"id": id, 
               "question": q, 
               "retrieve_docs": res} for id, q, res in zip(data.id, data.question, results)]

    # save
    save_path = os.path.join(save_dir, f"{split}.jsonl")
    with open(save_path, "w", encoding='utf-8') as f:
        for item in output:
            json.dump(item, f)
            f.write("\n")

