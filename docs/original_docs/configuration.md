# Configuration and Parameters

FlashRAG enables comprehensive management of various parameters for controlling the experiment. FlashRAG supports two types of parameter configurations: **YAML config file** and **parameter dict**. The parameters are assigned via the `Config` module.

In practice, `config` can be used in a dictionary access manner, such as:
```python
from flashrag.config import Config
config_dict = {'generator_model': 'llama2-7B'}
config = Config(config_dict=config_dict)
model_name = config['generator_model']
```

In most cases, the majority of parameters do not need to be modified. Specific application examples can be found below in the [Example Configuration](#example-configuration).

## Operation Logic

##### Config File

Config File should be in YAML format, and users should fill in the corresponding parameters according to the YAML syntax. In our library, we provide a template file for users to refer to, with comments explaining the specific meaning of each parameter.

The code for using a config file is as follows:
```python 
from flashrag.config import Config
config = Config(config_file_path='myconfig.yaml')
```

##### Parameter Dict

Another way is to set up the configuration via a Python dictionary, where keys are parameter names and values are parameter values. This method is more flexible compared to using a file.

The code for using a parameter dict is as follows:
```python 
from flashrag.config import Config
config_dict = {'generator_model': 'llama2-7B'}
config = Config(config_dict=config_dict)
```

##### Priority

In FlashRAG, we support combining both methods.

The priority of configuration methods is: Parameter Dict > Config File > Default Settings

The default settings are recorded in [basic_config.yaml](../flashrag/config/basic_config.yaml).

##### Path Settings

To facilitate the management of various model paths and index paths, we introduce `model2path`, `model2pooling`, and `method2index`. Users only need to fill in the corresponding names and paths here to automatically load the paths to be used.

For example, after setting the path of `llama2-7B` in `model2path`, you can directly specify it as the corresponding model without rewriting the path.
```yaml
model2path:
    llama2-7B: 'my_path'
generator_model: 'llama2-7B'
```

If the corresponding model path is not found in the dictionary, it will look for the parameter corresponding to the model path (e.g., `generator_model_path`).

This rule applies to the following parameters:
- retrieval_method
- rerank_model_name
- generator_model
- index_path

## Example Configuration

An example configuration file can be found in [basic_config.yaml](../flashrag/config/basic_config.yaml). Below we will go through each section in the configuration file.

```yaml
# basic settings

# ------------------------------------------------Global Paths------------------------------------------------#
# Paths to various models
model2path:
  e5: "intfloat/e5-base-v2"
  bge: "intfloat/e5-base-v2"
  contriever: "facebook/contriever"
  llama2-7B-chat: "meta-llama/Llama-2-7b-chat-hf"
  llama2-7B: "meta-llama/Llama-2-7b-hf"
  llama2-13B: "meta-llama/Llama-2-13b-hf"
  llama2-13B-chat: "meta-llama/Llama-2-13b-chat-hf"
  
# Pooling methods for each embedding model
model2pooling:
  e5: "mean"
  bge: "cls"
  contriever: "mean"
  jina: 'mean'
  dpr: cls

# Indexes path for retrieval models
method2index:
  e5: ~
  bm25: ~
  contriever: ~

# ------------------------------------------------Environment Settings------------------------------------------------#
# Directory paths for data and outputs
data_dir: "dataset/"
save_dir: "output/"

gpu_id: "0,1,2,3"
dataset_name: "nq" # name of the dataset in data_dir
split: ["test"]  # dataset split to load (e.g. train,dev,test)

# Sampling configurations for testing
test_sample_num: ~  # number of samples to test (only work in dev/test split), if None, test all samples
random_sample: False # whether to randomly sample the test samples

# Seed for reproducibility
seed: 2024

# Whether save intermediate data
save_intermediate_data: True
save_note: 'experiment'

# -------------------------------------------------Retrieval Settings------------------------------------------------#
# If set the name, the model path will be find in global paths
retrieval_method: "e5"  # name or path of the retrieval model. 
retrieval_model_path: ~ # path to the retrieval model
index_path: ~ # set automatically if not provided. 
faiss_gpu: False # whether use gpu to hold index
corpus_path: ~  # path to corpus in '.jsonl' format that store the documents

use_sentence_transformer: False # If set, the retriever will be load through `sentence transformer` library
retrieval_topk: 5 # number of retrieved documents
retrieval_batch_size: 256  # batch size for retrieval
retrieval_use_fp16: True  # whether to use fp16 for retrieval model
retrieval_query_max_length: 128  # max length of the query
save_retrieval_cache: True # whether to save the retrieval cache
use_retrieval_cache: False # whether to use the retrieval cache
retrieval_cache_path: ~ # path to the retrieval cache
retrieval_pooling_method: ~ # set automatically if not provided

use_reranker: False # whether to use reranker
rerank_model_name: ~ # same as retrieval_method
rerank_model_path: ~ # path to reranker model, path will be automatically find in `model2path`
rerank_pooling_method: ~
rerank_topk: 5  # number of remain documents after reranking
rerank_max_length: 512 
rerank_batch_size: 256 # batch size for reranker
rerank_use_fp16: True

# -------------------------------------------------Generator Settings------------------------------------------------#
framework: fschat # inference frame work of LLM, supporting: 'hf','vllm','fschat', 'openai'
generator_model: "llama3-8B-instruct" # name or path of the generator model
# setting for openai model, only valid in openai framework
openai_setting:
  api_key: ~
  base_url: ~

generator_model_path: ~
generator_max_input_len: 1024  # max length of the input
generator_batch_size: 4 # batch size for generation, invalid for vllm
generation_params:  
  #do_sample: false
  max_tokens: 32
  #temperature: 1.0
  #top_p: 1.0
use_fid: False # whether to use FID, only valid in encoder-decoder model

# -------------------------------------------------Evaluation Settings------------------------------------------------#
# Metrics to evaluate the result
metrics: ['em','f1','acc,'precision','recall'] 
# Specify setting for metric, will be called within certain metrics
metric_setting: 
  retrieval_recall_topk: 5
save_metric_score: True #　whether to save the metric score into txt file
```


### Global Settings

Here the paths to models, indexes, and pooling methods for each embedding model are saved. Later, you only need to specify the corresponding name to automatically load the path.

```yaml
# Paths to various models
model2path:
  e5: "intfloat/e5-base-v2"
  bge: "intfloat/e5-base-v2"
  contriever: "facebook/contriever"
  llama2-7B-chat: "meta-llama/Llama-2-7b-chat-hf"
  llama2-7B: "meta-llama/Llama-2-7b-hf"
  llama2-13B: "meta-llama/Llama-2-13b-hf"
  llama2-13B-chat: "meta-llama/Llama-2-13b-chat-hf"
  
# Pooling methods for each embedding model
model2pooling:
  e5: "mean"
  bge: "cls"
  contriever: "mean"
  jina: 'mean'
  dpr: cls

# Indexes path for retrieval models
method2index:
  e5: ~
  bm25: ~
  contriever: ~
```

### Environment Settings

Here mainly manage various configurations of the experiment.

```yaml
# Directory paths for data and outputs
data_dir: "dataset/"
save_dir: "output/"

gpu_id: "0,1,2,3"
dataset_name: "nq" # name of the dataset in data_dir
split: ["test"]  # dataset split to load (e.g. train,dev,test)

# Sampling configurations for testing
test_sample_num: ~  # number of samples to test (only work in dev/test split), if None, test all samples
random_sample: False # whether to randomly sample the test samples

# Seed for reproducibility
seed: 2024

# Whether save intermediate data
save_intermediate_data: True
save_note: 'experiment'
```

- `split`: Specifies the dataset split to load, multiple splits can be specified and used separately in the code.
- `save_note`, `save_dir`: Each experiment will create a new folder in `save_dir`, and add `save_note` as a marker in the folder name.
- `save_intermediate_data`: If enabled, intermediate results will be recorded in the experiment folder, including the retrieval content, generated content, and corresponding evaluation metrics for the content.
- `gpu_id`: Specifies which GPUs to use for the experiment, supports multiple GPUs.

If you do not want to use the entire dataset for testing, you can set 'test_sample_num' (invalid for the training set). If set, the corresponding number of samples at the front will be selected. If 'random_sample' is enabled, it will be randomly selected.

### Retrieval Settings

This section manages various parameters for the retriever and reranker.

```yaml
retrieval_method: "e5"  # name or path of the retrieval model. 
retrieval_model_path: ~ # path to the retrieval model
index_path: ~ # set automatically if not provided. 
faiss_gpu: False # whether use gpu to hold index
corpus_path: ~  # path to corpus in '.jsonl' format that store the documents

use_sentence_transformer: False # If set, the retriever will be load through `sentence transformer` library
retrieval_topk: 5 # number of retrieved documents
retrieval_batch_size: 256  # batch size for retrieval
retrieval_use_fp16: True  # whether to use fp16 for retrieval model
retrieval_query_max_length: 128  # max length of the query
save_retrieval_cache: True # whether to save the retrieval cache
use_retrieval_cache: False # whether to use the retrieval cache
retrieval_cache_path: ~ # path to the retrieval cache
retrieval_pooling_method: ~ # set automatically if not provided

use_reranker: False # whether to use reranker
rerank_model_name: ~ # same as retrieval_method
rerank_model_path: ~ # path to reranker model, path will be automatically find in `model2path`
rerank_pooling_method: ~
rerank_topk: 5  # number of remain documents after reranking
rerank_max_length: 512 
rerank_batch_size: 256 # batch size for reranker
rerank_use_fp16: True
```


If the paths in the previous dictionary are filled, only `retrieval_method` and `corpus_path` need to be modified (no need to modify `index_path` and `retrieval_model_path`).

FlashRAG supports saving and reusing retrieval results. When reusing, it will look in the cache to see if there is a query identical to the current one and read the corresponding results.
- `save_retrieval_cache`: If set to `True`, it will save the retrieval results as a JSON file, recording the retrieval results and scores for each query, enabling reuse next time.
- `retrieval_cache_path`: Set to the path of the previously saved retrieval cache.

To use a reranker, set `use_reranker` to `True` and fill in `rerank_model_name`. For Bi-Embedding type rerankers, the pooling method needs to be set, similar to the retrieval method.

If set `use_sentence_transformer` to `True`, there is no need to set consider pooling method.

### Generator Settings

This section records various settings for the generator.

```yaml
framework: fschat # inference frame work of LLM, supporting: 'hf','vllm','fschat', 'openai'
generator_model: "llama3-8B-instruct" # name or path of the generator model
# setting for openai model, only valid in openai framework
openai_setting:
  api_key: ~
  base_url: ~
generator_model_path: ~
generator_max_input_len: 1024  # max length of the input
generator_batch_size: 4 # batch size for generation, invalid for vllm
generation_params:  
  max_tokens: 32
use_fid: False # whether to use FID, only valid in encoder-decoder model
```

- `framework`: The base framework of the generator. It is recommended to use `vllm` for deployment.
- `generation_params`: Parameters needed during generation. The parameter names may need to be adjusted according to different frameworks. Refer to the function descriptions of vllm or huggingface generation for details.

### Evaluation Settings

This section sets various settings used during evaluation. If you use a custom evaluation metric, you can add your parameters in `metric_setting` and call them in the metric.

```yaml
# Metrics to evaluate the result
metrics: ['em','f1','acc','precision','recall'] 
# Specify setting for metric, will be called within certain metrics
metric_setting: 
  retrieval_recall_topk: 5
save_metric_score: True #　whether to save the metric score into txt file
```

- `metrics`: The specific evaluation metrics to be used. The values are the `metric_name` of the evaluation metrics. Currently supported evaluation metrics can be found [<u>here</u>](../flashrag/evaluator/metrics.py).