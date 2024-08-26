## 1. Overview

This document aims to introduce various configurations and functionalities of this project using the Standard RAG process as an example. Additional documentation will be provided later to cover complex uses such as reproducing existing methods and detailed usage of individual components.

The Standard RAG process includes the following three steps:
1. Retrieve relevant documents from the knowledge base based on the user's query.
2. Incorporate the retrieved documents and the original query into a prompt.
3. Input the prompt into the generator.

This document will demonstrate the RAG process using `E5` as the retriever and `Llama2-7B-Chat` as the generator.

## 2. Prerequisites

To smoothly run the entire RAG process, you need to complete the following five preparations:

1. Install the project and its dependencies.
2. Download the required models.
3. Download the necessary datasets (a [toy dataset](../examples/quick_start/dataset/nq) is provided).
4. Download the document collection for retrieval (a [toy corpus](../examples/quick_start/indexes/general_knowledge.jsonl) is provided).
5. Build the index for retrieval (a [toy index](../examples/quick_start/indexes/e5_Flat.index) is provided).

To save time in getting started, we provide toy datasets, document collections, and corresponding indices. Therefore, you only need to complete the first two steps to successfully run the entire process.

### 2.1 Installing the Project and Dependencies

Install the project and its dependencies using the following commands.

Note that if you encounter issues installing the `vllm`, `fschat`, or `pyserini` packages, you can comment them out in the `requirement.txt` file. These packages are required for certain functionalities, but omitting them temporarily won't affect the workflow described in this document.

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e . 
```

### 2.2 Download Models

You need to download the following two models:

- E5-base-v2
- Llama2-7B-Chat

You can download the models from [Huggingface](https://huggingface.co/intfloat/e5-base-v2). If you are in China, it's recommended to use the mirror platform [hf-mirror](https://hf-mirror.com/) for downloading.

### 2.3 Download Datasets

The datasets include queries and corresponding standard answers, allowing us to evaluate the effectiveness of the RAG system.

For simplicity, we have sampled 17 pieces of data from NQ as a toy dataset, located at [examples/quick_start/dataset/nq](../examples/quick_start/dataset/nq/). The subsequent RAG process will be conducted on these questions.

Our repository also provides a large number of processed benchmark datasets. You can visit our  [huggingface datasets](https://huggingface.co/datasets/ignore/FlashRAG_datasets) to download and use them.

### 2.4 Downloading the Document Collection

The document collection contains a large number of segmented paragraphs, serving as the external knowledge source for the RAG system. Since commonly used document collections are often very large (~5G or more), we use a [general knowledge dataset](https://huggingface.co/datasets/MuskumPillerum/General-Knowledge) as a toy collection, located at  [examples/quick_start/indexes/general_knowledge.jsonl](../examples/quick_start/indexes/general_knowledge.jsonl)。

> Due to the small number of documents, many queries may not find relevant texts, which could affect the final retrieval results.


If you need to obtain the full document collection, you can visit our [huggingface dataset](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) to download and use them.


### 2.5 Building the Retrieval Index

To improve retrieval efficiency, we often need to build the retrieval index in advance. For the BM25 method, the index is usually an inverted index (a directory in our project). For various embedding methods, the index is a Faiss database containing the embeddings of all texts in the document collection (an .index file). **Each index corresponds to a corpus and a retrieval method**, meaning that every time you want to use a new embedding model, you need to rebuild the index.

Here, we provide a [toy index](../examples/quick_start/indexes/e5_Flat.index), built using E5-base-v2 and the aforementioned toy corpus.

If you want to use your own retrieval model and documents, you can refer to our [index building document](./building-index.md) to build your index.


## 3. Running the RAG Process

In the following steps, we will break down each step and demonstrate the corresponding code. The complete code will be provided at the end, or you can refer to the [simple_pipeline.py](../examples/quick_start/simple_pipeline.py) file.


### 3.1 Loading the Config

First, we need to load the `Config` and fill in the paths of the previously downloaded items.

`Config` manages all the paths and hyperparameters in the experiment. In FlashRAG, various parameters can be passed into the Config via a yaml file or a Python dictionary. The passed parameters will replace the default internal parameters. For detailed parameter information and their default values, you can refer to our [`basic_config.yaml`](../flashrag/config/basic_config.yaml).


Here, we directly pass the paths via a dictionary.

```python
from flashrag.config import Config

config_dict = { 
    'data_dir': 'dataset/',
    'index_path': 'indexes/e5_Flat.index',
    'corpus_path': 'indexes/general_knowledge.jsonl',
    'model2path': {'e5': <retriever_path>, 'llama2-7B-chat': <generator_path>},
    'generator_model': 'llama2-7B-chat',
    'retrieval_method': 'e5',
    'metrics': ['em', 'f1', 'acc'],
    'retrieval_topk': 1,
    'save_intermediate_data': True
}

config = Config(config_dict=config_dict)
```

### 3.2 Loading the Dataset and Pipeline

Next, we need to load the dataset and pipeline.

The dataset can be automatically read through the previously set config; we only need to select the corresponding test set.

The pipeline loading requires us to select an appropriate pipeline based on the desired RAG process. Here, we choose `SequentialPipeline` for the Standard RAG process mentioned earlier.
The pipeline will automatically load the corresponding components (retriever and generator) and complete various initializations.

```python
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline

all_split = get_dataset(config)
test_data = all_split['test']
pipeline = SequentialPipeline(config)
```


### 3.3 Running the RAG Process

After completing the above steps, we only need to call the pipeline's `.run` method to run the RAG process on the dataset and generate evaluation results. This method returns a dataset containing intermediate results and final results, with the pred attribute containing the model's predictions.

Note that because we provided toy document collections and indices, the results might be relatively poor. Consider using your own document collections and indices for better results.

After the process completes, all results will be saved in a folder corresponding to the current experiment, including the retrieval and generation results for each query, overall evaluation scores, and more.

The complete code is as follows:

```python
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline

config_dict = { 
                'data_dir': 'dataset/',
                'index_path': 'indexes/e5_Flat.index',
                'corpus_path': 'indexes/general_knowledge.jsonl',
                'model2path': {'e5': <retriever_path>, 'llama2-7B-chat': <generator_path>},
                'generator_model': 'llama2-7B-chat',
                'retrieval_method': 'e5',
                'metrics': ['em','f1','acc'],
                'retrieval_topk': 1,
                'save_intermediate_data': True
            }

config = Config(config_dict = config_dict)

all_split = get_dataset(config)
test_data = all_split['test']
pipeline = SequentialPipeline(config)

output_dataset = pipeline.run(test_data,do_eval=True)
print("---generation output---")
print(output_dataset.pred)
```

