## 1. Overview

本文旨在以Standard RAG流程为例进行介绍本项目所涉及的各种配置和功能。关于复现已有方法以及各个组件的详细用法等复杂用法，后续会有额外的文档来介绍。


Standard RAG的流程包括以下三个步骤: 
1. 根据用户query从知识库种检索一些相关文档
2. 将检索结果和原始query放入prompt
3. 将prompt输入生成器中

本文将实现以``E5``作为检索器， ``Llama2-7B-Chat``作为生成器的RAG流程。

## 2. 前置准备

为了顺利运行整个RAG流程，需要完成以下五项准备工作:

1. 安装本项目以及对应的依赖库
2. 下载需要的各种模型
3. 下载需要的数据集 (已提供[toy dataset](../examples/quick_start/dataset/nq))
4. 下载用于检索的文档集合(已提供[toy corpus](../examples/quick_start/indexes/general_knowledge.jsonl))
5. 构建用于检索的index (已提供[toy index](../examples/quick_start/indexes/e5_Flat.index))


为了能够节省入门所需要的时间，我们提供了玩具数据集、文档集合以及对应的index。因此实际上只需要进行前两步就可以顺利完成整个流程。

### 2.1 安装本项目以及对应的依赖库

本项目以及对应依赖库的安装通过如下命令来完成。

需要注意的是，如果在```vllm```,```fschat```,```pyserini```包的安装上存在问题，可以先将```requirement.txt```中对应的包名注释掉。安装这些包只是为了某些功能的完整性，去掉之后暂时不影响本文流程的运行。

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e . 
```

### 2.2 下载模型

总共需要下载以下两个模型:

- E5-base-v2
- Llama2-7B-Chat

模型的下载可以通过[Huggingface](https://huggingface.co/intfloat/e5-base-v2)。如果是中国用户，推荐使用镜像平台[hf-mirror](https://hf-mirror.com/)进行下载。

### 2.3 下载数据集

数据集包含了query和对应的标准答案，能够让我们评估构建的RAG系统的效果。

为了简便，我们从NQ中采样了17条数据作为toy dataset, 其地址在 [examples/quick_start/dataset/nq](../examples/quick_start/dataset/nq/)。后续RAG流程将在这些questions上进行。

我们的仓库中同样提供了大量处理好的基准数据集，可以访问我们[huggingface上的数据集](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)进行下载和使用。

### 2.4 下载文档集合

文档集合包含了大量的切分好的段落，是RAG系统的外部知识来源。由于常用的文档集合往往非常大(~5G以上)，我们使用了一个通用知识数据集作为检索文档， 地址为 [examples/quick_start/indexes/general_knowledge.jsonl](../examples/quick_start/indexes/general_knowledge.jsonl)。
> 由于文档数量较少，可能很多query都无法搜到相关的文本，这可能会影响最终的检索结果。


如果需要获取完整的文档集合，可以访问我们[huggingface上的数据集](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus)进行下载和使用。


### 2.5 构建用于检索的索引

为了提高检索的查询效率，我们往往需要提前构建检索的索引。对于BM25方法，索引往往是倒排表(在我们项目中是一个文件夹)。对于各类embedding方法，索引是一个包含检索文档集合中所有文本的embedding的faiss数据库(一个.index文件)。**每个索引对应着一个corpus和一种检索方式**，也就是每当想使用一种新的embedding模型，都得重新构建索引。

在这里我们提供了一个[toy index](../examples/quick_start/indexes/e5_Flat.index)，其使用E5-base-v2以及前面的toy corpus进行构建。

如果想使用自己的检索模型和检索文档，可以参考我们的[索引构建文档](./building-index.md)来构建。


## 3. 运行RAG流程

我们在下面的步骤中会依次拆解各个步骤，并展示相应的代码。完整的代码会在最后放出，或者可以参考[simple_pipeline.py](../examples/quick_start/simple_pipeline.py)文件。


### 3.1 加载Config

首先我们需要加载``Config``，并填入之前下载的各种东西的路径。

`Config`管理着实验中所有的路径以及超参数。在FlashRAG中，各种参数可以通过`yaml`文件或者python内的字典传入Config中。传入的参数会替换默认的内部参数，具体的各种参数以及其默认值可以参考我们的[`basic_config.yaml`](../flashrag/config/basic_config.yaml)。

在这里我们直接通过字典传入各种路径。

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

### 3.2 加载数据集以及Pipeline

接着，我们需要加载相应的数据集和pipeline。

数据集可以通过前面设置的config自动读取，我们只需要选择对应需要的test集合就可以了。

而pipeline的加载需要我们先根据想进行的RAG流程选择一个合适的pipeline。这里我们选择`SequentialPipeline`，用于进行前面提到的Standard RAG流程。
然后pipeline的过程中会自动加载对应的组件(检索器和生成器)，并完成各种初始化。

```python
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline

all_split = get_dataset(config)
test_data = all_split['test']
pipeline = SequentialPipeline(config)
```


### 3.3 运行RAG流程

在完成上面的步骤之后，我们只需要调用`pipeline`的`.run`方法即可在数据集上运行RAG流程并生成评估结果。该方法返回一个包含中间结果和运行结果的数据集，其中`pred`属性的内容则是模型的预测结果。

需要注意的是，由于前面我们提供的是玩具文档集合和索引，因此结果可能会比较差。可以考虑换成自己的文档集合和索引，应该能够取得较好的结果。

在运行结束后，所有运行结果会被单独保存在本次实验对应的文件夹内，包括每个query的检索和生成结果，整体的评价分数等。

完整的代码如下:
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
