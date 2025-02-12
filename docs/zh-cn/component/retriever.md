# 检索器

检索器（Retriever）是一个用于从大量文档或数据中高效检索相关信息的组件。其核心功能是根据用户输入的查询（如文本或图像），从预先构建的文档库中检索出最相关的文档或信息。 检索器的工作原理是通过计算查询和文档之间的相似度，并返回最相关的文档。

检索器的工作流程通常包括以下几个步骤：

1. 输入查询：用户提交查询，查询可以是自然语言文本、图像或两者的组合（即多模态查询），检索器会对查询进行词项拆解（稀疏检索器）或者嵌入表示生成（稠密检索器）
2. 文档索引：系统通过预先构建的索引（如 BM25 或深度学习嵌入的索引）将文档内容转换为机器可理解的向量或特征表示。
3. 相似度计算：根据查询和文档的相似度（如基于词频、嵌入向量等方法），检索器从文档库中找出与查询最相关的文档。
4. 排序与返回：根据相关性分数对结果进行排序，并返回给用户最匹配的文档。

## 支持的检索器列表

FlashRAG支持以下几种检索器：

1. BM25检索器: 支持两种后端BM25算法实现：bm25s(基于BM25算法的实现，速度更快)、pyserini(基于Pyserini的实现，更稳定通用，需安装Java)。

2. 稠密检索器: 支持Bert结构的检索器, T5结构的检索器以及sentence-transformers支持的各类检索器。
    - Bert结构检索器: 包括但不仅限于 DPR、Contriever、E5、BGE、JINA、GTE等模型。
    - T5结构检索器，如ANCE等基于T5-Encoder的模型。

> [!TIP] 如何查看某个稠密检索器是否支持?
> 通常来说，如果模型支持`sentence-transformers`框架，则可以直接使用。否则则只需打开模型目录下的`config.json`文件，查看里面的`architectures`参数。如果为`BertModel`或者类`Bert`的结构，则可以使用。


3. 多模态检索器: 支持Openai系列的Clip模型以及类Clip结构的其他模型。
    - Openai系列的Clip模型: 包括但不仅限于`openai/clip-vit-large-patch14`, `openai/clip-vit-base-patch32`，以及类似结构的其他模型。
    - Clip架构的其他模型: `jinaai/jina-clip-v2`, `OFA-Sys/chinese-clip-vit-base-patch16` 等Clip结构的其他模型。

> [!NOTE] 由于Clip结构的模型众多，这里未能全部列出，可以直接尝试模型是否能够加载或者查看`retriever/encoder.py`中的`ClipEncoder`是否支持该模型。后续会添加对`open-clip`相关模型的支持。


## 检索器的配置与加载

检索器的配置可以参考[配置文件相关说明](../configuration/retriever.md)。

如需单独使用检索器，只需使用下面的代码进行加载：
```python
from flashrag.config import Config
from flashrag.utils import get_retriever

# 加载自己的config，配置好检索器的参数
config = Config('my_config.yaml')
retriever = get_retriever(config)
```

## 使用

FlashRAG中的检索器分为三个部分:
* 纯文本模态检索器
* 多模态检索器
* 多路检索器 (多个任意检索器的组合)

三个部分的使用逻辑和输入输出参数基本一致，具体使用方式如下。

### 纯文本模态检索器

纯文本模态的检索器包含两个检索接口:

- `search(query: str, top_k: int = 10, return_score: bool = False)`
- `batch_search(query: list, top_k: int = 10, return_score: bool = False)`

两个接口的输入参数与处理逻辑完全一样，但`batch_search`能够同时处理多个query。**推荐在所有情况下使用`batch_search`接口。**

#### 输入参数

`search`和`batch_search`接受下面的参数作为输入:

- **查询文本**：待检索的文本，可以是一个简单的自然语言问题或描述。输入类型可以是字符串或字符串列表。
  
- **返回结果数（topk）**：指定返回最相关的前K个结果。默认值为5。

- **是否返回分数（return_score）**：指定是否返回每个结果的相关性得分，开启后默认为`False`。开启后函数的返回为元组，第一个元素为检索结果，第二个元素为对应的相关性得分。

> [!NOTE] 接口中`topk`参数与Config中配置的`retrieval_topk`含义相同，接口中的优先级更高。

#### 返回结果

`search`接口的输出包含检索结果和相关性分数两部分:

- 检索结果: 其格式为`List[dict]`, 每个`dict`存放的是检索corpus中一个文档的信息，包括`id`, `contents`等，与corpus中的内容相对应。
- 相关性分数: 其格式为`List[float]`, 每个分数表示query与对应位置文档计算出的相关性分数。

对于`batch_search`接口，上述输出外面会再嵌套一层列表对应batch中每个query的信息，即:
- 检索结果: 格式为`List[List[dict]]`
- 相关性分数: 格式为`List[List[float]]`

如果打开了`return_score`， 返回值格式为元组: `(retrieval_result, doc_scores)`,否则返回值为列表: `retrieval_result`。



### 多模态检索器

多模态检索器支持同时检索文本和图像两种模态的数据。它通过预构建的Faiss索引实现高效的查询，支持图像和文本两种输入类型。

多模态检索器包含两个检索接口：

- `search(query: str, target_modal: str = "text", top_k: int = 10, return_score: bool = False)`
- `batch_search(query: list, target_modal: str = "text", top_k: int = 10, return_score: bool = False)`

这两个接口的输入参数和处理逻辑一致，但`batch_search`可以同时处理多个查询。**建议在所有情况下使用`batch_search`接口。**

#### 输入参数

`search`和`batch_search`接受以下输入参数：

- **查询内容**：待检索的内容。对于文本模态，可以直接输入字符串。对于图像模态，支持输入图片路径、`PIL.Image`格式的图片对象以及图片URL。

> [!WARNING]
> 不支持混合模态的输入，单次输入的query_list必须为同一模态的数据。
  
- **目标模态（target_modal）**：指定查询的模态。支持的值为`"text"`和`"image"`，分别代表文本和图像。默认值为`"text"`。检索器会计算query与指定目标模态的数据的相似度。

- **返回结果数（topk）**：指定返回最相关的前K个结果。默认值为5。

- **是否返回分数（return_score）**：指定是否返回每个结果的相关性得分，开启后返回一个包含检索结果和相关性分数的元组。默认值为`False`。

#### 返回结果

多模态检索器的输出格式与文本模态相同。


### 多路检索器

多路检索器支持对任意多个检索器进行组合(**包括文本检索器和多模态检索器的组合**)，实现多路检索和聚合。为了简化使用，多路检索器的加载方法和使用方法与单个检索器一致。

#### 输入参数

多路检索器的输入参数与多模态检索器相同。其`search`和`batch_search`接口接受以下输入参数：

- **查询内容**：待检索的内容。对于文本模态，可以直接输入字符串。对于图像模态，支持输入图片路径、`PIL.Image`格式的图片对象以及图片URL。

> [!TIP]
> 这里支持混合模态的输入，可以混合输入图片类型的query和文本类型的query。
  
- **目标模态（target_modal）**：指定查询的模态。支持的值为`"text"`和`"image"`，分别代表文本和图像。默认值为`"text"`。检索器会计算query与指定目标模态的数据的相似度。

- **返回结果数（topk）**：指定返回最相关的前K个结果。默认值为5。

- **是否返回分数（return_score）**：指定是否返回每个结果的相关性得分，开启后返回一个包含检索结果和相关性分数的元组。默认值为`False`。

#### 返回结果

多路检索器的返回结果格式与单个检索器一致。但是由于多路检索器可能涉及多个corpus，每个corpus内的文档存储不一样，所以最终返回的检索文档(每个dict)格式可能并不一致。

#### 多路聚合方式

多路检索器支持多种聚合方式，包括：
- concat: 把所有结果直接拼接进行返回。
- rrf: 基于倒数排名融合算法进行聚合，具体可以参考[RRF](https://learn.microsoft.com/zh-cn/azure/search/hybrid-search-ranking)
- rerank: 基于一个reranker对检索文档进行重新排序，仅支持query均为纯文本的场景。

#### 使用方式

##### 例子一: 纯文本检索器 + 纯文本检索器

分别使用BM25和E5各检索10个文档，使用bce-reranker进行聚合，保留最终top5。

```yaml
"multi_retriever_setting": {
    "merge_method": "rerank",
    "topk": 5,
    'rerank_model_name': 'bge-reranker',
    'rerank_model_path': 'model/bce-reranker-base_v1',
    "retriever_list": [
        {
            "retrieval_method": "bm25",
            "corpus_path": "general_knowledge/general_knowledge.jsonl",
            "index_path": "general_knowledge/bm25",
            "retrieval_topk": 10,
            "bm25_backend": "pyserini",
        },
        {
            "retrieval_method": "e5",
            "corpus_path": "general_knowledge/general_knowledge.jsonl",
            "index_path": "general_knowledge/e5_Flat.index",
            "retrieval_topk": 10,
        },
        
    ],
}
```

##### 例子二: 纯文本检索器 + 多模态检索器

使用BM25对文本query进行文本模态的检索 + 使用Clip对图像query进行图像模态的检索

```yaml
"multi_retriever_setting": {
    "merge_method": "concat",
    "retriever_list": [
        {
            "retrieval_method": "bm25",
            "corpus_path": "datasets/mmqa/train.parquet",
            "index_path": "indexes/mmqa/bm25",
            "retrieval_topk": 1,
            "bm25_backend": "pyserini",
        },
        {
            "retrieval_method": "openai-clip-336",
            "corpus_path": "mmqa/train.parquet",
            "retrieval_model_path": "model/openai-clip-vit-large-patch14",
            "multimodal_index_path_dict": {
                "image": "indexes/mmqa/openai-clip-vit-large-patch14_Flat_image.index",
                "text": "indexes/mmqa/openai-clip-vit-large-patch14_Flat_text.index",
            },
            "retrieval_topk": 1,
        },
    ],
}
```


使用代码:

```python
config = Config("my_config.yaml", config_dict=config_dict)
retriever = get_retriever(config)
query_list = ['test_pic.png', 'who is the president of USA?']
output, score = retriever.batch_search(query_list, target_modal='text', return_score=True)
```
