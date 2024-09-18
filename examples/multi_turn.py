"""
A simple case to use generator for multi-turn interaction
"""
import argparse
from flashrag.config import Config
from flashrag.utils import get_generator, get_dataset, get_retriever
from flashrag.prompt import PromptTemplate

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
args = parser.parse_args()
config_dict = {
    "data_dir": "dataset/",
    "model2path": {"llama3-8B-instruct": args.model_path},
    "generator_model": "llama3-8B-instruct",
    "retrieval_method": "e5",
    "metrics": ["em", "f1", "acc"],
    "retrieval_topk": 1,
    "save_intermediate_data": True,
}

config = Config(config_dict=config_dict)
generator = get_generator(config)
prompt_template = PromptTemplate(config)

messages = [{"role":"system", "content":"You are a helpful assistant, please follow the user's instructions to complete the task."}]

### turn 1
messages.append({"role": "user", "content":"Who is the wife of the current US President?"})
print(messages)
input_prompt = prompt_template.get_string(messages=messages)
print(f"#### Turn 1 input: {input_prompt}")
output = generator.generate(input_prompt)[0]
print(f"#### Turn 1 output: {output}")

### turn 2
messages.append({"role": "system", "content": output})
# add new input here
messages.append({"role": "user", "content": "Your answer is incorrect, can you provide a high-quality answer again?"})
input_prompt = prompt_template.get_string(messages=messages)
print(f"#### Turn 2 input: {input_prompt}")
output = generator.generate(input_prompt)[0]
print(f"#### Turn 2 output: {output}")

