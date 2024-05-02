# Usage 

## Build index

```python
from flashrag.config import Config
from flashrag.retriever import Index_Builder

config = Config('my_config.yaml')
index_builder = Index_Builder(config)
index_builder.build_index()
```

## Run naive RAG pipeline

```python
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import NaiveRAG

config = Config('my_config.yaml')
all_split = get_dataset(config)
test_data = all_split['test']
pipeline = NaiveRAG(config)
result = pipeline.run(test_data)
print(result)  # {"em": ..., "f1":...}
```

# Main Classes

There are five key classes in RAG framework: `Config`, `Dataset`, `Retriever`, `Generator` and `Evaluator`.

## Config

`Config` class supports using `.yaml` files as input or variables as input. The priority of variables is higher than that of files. **All subsequent component settings depend on `Config`.**

If there are variables that need to be used that are not specified in these two places, default values will be loaded (`basic_config.yaml`).

```python
from flashrag.config import Config

config_dict = {"retrieval_method": "bge"}
config = Config('my_config.yaml', config_dict = config_dict)
print(config['bge'])
```

## Dataset

The original file of the dataest is a ```.jsonl```, each line is a dictionary in the following form:
```json
{
    "id": "train_0", 
    "question": "who's hosting the super bowl in 2019", 
    "golden_answers":["Atlanta, Georgia"],
    "metadata":{}
}
```

Inside `dataset` class, 

## Retriever

```python
from flashrag.config import Config
from flashrag.utils import get_dataset, get_retriever

config = Config()
retriever = get_retriever(config)
query = "Australian Dance awards celebration"
results = retriever.search(query = query, num = 5)
print(results)
```

## Generator

```python
from flashrag.config import Config
from flashrag.utils import get_dataset, get_generator

config = Config()
generator = get_generator(config)
query = "Who got the first nobel prize in physics?"
output = generator.generate(query)
print(output)
```

## Evaluator

```python
config = Config()
evaluator = Evaluator(config)

all_split = get_dataset(config)
test_data = all_split['test']

test_data = test_data.data
for item in test_data:
    item['output'] = {"pred": item['question']}

eval_result = evaluator.evaluate(test_data)
print(eval_result)
```