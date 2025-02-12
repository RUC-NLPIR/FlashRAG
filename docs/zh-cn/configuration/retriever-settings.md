# 检索器设置

检索器相关的参数主要包含四部分: 
* 基本检索模型参数
* 检索涉及的超参数
* 重排器相关参数
* 多路检索功能参数

### 基本检索模型参数

> [!TIP] 
> 如果已经在前面字典中填充好了`model2path`和`method2index`，仅需修改 `retrieval_method` 和 `corpus_path`即可使用检索功能。


- **retrieval_method**  
  指定检索模型的自定义名称。

- **retrieval_model_path**  
  设置检索模型的路径。如果没有指定路径，系统会自动从`model2path`查找路径。

- **index_path**  
  指定检索索引的路径。如果没有提供，系统会自动从`methodwindex`设置索引路径。

- **corpus_path**  
  设置包含文档的语料库路径，文件格式应为 `.jsonl`。该文件包含用于检索的所有文档。

- **faiss_gpu**  
  指定是否使用 GPU 存储和处理索引。如果设置为 `True`，则检索过程会使用 GPU 进行加速，适用于大规模数据。使用GPU会占用每张显卡的部分显存，可能导致GPU OOM的情况，推荐设置为`False`。

- **multimodal_index_path_dict**  
  用于多模态检索。该参数是一个字典，指定文本和图像等不同模态的索引路径。例如，`{'text': 'path/to/text_index', 'image': 'path/to/image_index'}`，其中可以设置为 `None` 表示该模态不使用索引。

- **retrieval_pooling_method**  
  设置检索结果的池化方法，如未指定则自动设置。如果将 `use_sentence_transformer` 设置为 `True`，则无需设置池化方法。支持的池化方法包括: `cls`, `pooler`, `mean`。

- **use_sentence_transformer**
  是否使用`sentence_transformer`框架进行检索器加载。使用前需确保模型支持`sentence_transformer`。

- **bm25_backend**  
  设置使用的 BM25 后端。可选的有 `pyserini` 和 `bm25s`，用于文档排序和匹配度计算。


FlashRAG 支持保存和重用检索结果。在重用时，它会查看缓存中是否有与当前查询相同的记录，并读取相应的结果。
- `save_retrieval_cache`：如果设置为 `True`，将会把检索结果保存为 JSON 文件，记录每个查询的检索结果和得分，方便下次重用。
- `retrieval_cache_path`：设置为之前保存的检索缓存的路径。

- **save_retrieval_cache**  
  设置是否保存检索缓存。如果为 `True`，检索结果会被缓存并保存为文件，以便下次使用时避免重复计算。

- **use_retrieval_cache**  
  设置是否使用已保存的检索缓存。如果为 `True`，系统会从缓存中加载结果，避免重复执行检索操作。

- **retrieval_cache_path**  
  设置检索缓存文件的路径。该路径指定了保存或加载检索缓存的位置。

- **retrieval_pooling_method**  
  设置检索结果的池化方法。池化方法决定了如何从多个候选文档中选择最相关的结果，若未指定则自动设置。



### 检索涉及的超参数

- **instruction**  
  部分检索器(比如E5,BGE)训练时需要在query/doc前拼接固定的指令，需要在这里填入拼接在query前面的instruction。对部分常用模型会自动补充。

- **retrieval_topk**  
  设置每次检索时返回的文档数量。例如，设置为 `5` 表示返回与查询最相关的前五篇文档。

- **retrieval_batch_size**  
  设置检索时的批次大小。批量检索可以加速处理大规模查询。通常，较大的批次会提高效率，但也可能增加内存占用。

- **retrieval_use_fp16**  
  指定是否使用 FP16 精度进行检索模型的计算。使用 FP16 可以加速计算，特别是在使用 GPU 时。

- **retrieval_query_max_length**  
  设置查询的最大长度。该值控制查询中可包含的最大字符数，确保检索过程不会超出模型的处理限制。


### 重排器相关参数

要使用重新排序器，将 `use_reranker` 设置为 `True` 并填写 `rerank_model_name`。FlashRAG支持使用两类重排器: `Bi-Encoder`和`Cross-Encoder`模型。

* `Bi-Encoder`模型即各类embedding模型，其作为重排器的设置方法和retrieval的设置方法相同，需要设置`pooling_method`。

* `Cross-Encoder`模型包括bge-reranker，gte-reranker等模型，只需设置各类超参数即可使用。


- **use_reranker**  
  是否启用重新排序器。启用后，在初步检索后，系统将再次对检索结果进行排序，提升检索的精度。

- **rerank_model_name**  
  设置重新排序模型的名称。该模型将在检索后对结果进行排序。

- **rerank_model_path**  
  设置重新排序模型的路径。该路径指定了用于排序的模型位置。

- **rerank_pooling_method**  
  设置重新排序时使用的池化方法。池化方法决定了如何处理多个检索结果并选择最相关的。

- **rerank_topk**  
  设置重新排序后保留的文档数量，可以与检索阶段设置的topk搭配使用(比如检索top100+重排保留top10)。

- **rerank_max_length**  
  设置重新排序时处理的最大的文本长度(query+doc)，较长的文本会被截断以适应模型的处理限制。

- **rerank_batch_size**  
  设置重新排序时的批次大小，控制每次处理多少文档。

- **rerank_use_fp16**  
  设置是否使用 FP16 精度进行重新排序计算。这可以加速排序过程，尤其是在使用 GPU 时。


### 多路检索相关设置


FlashRAG支持多路检索，即使用多个检索器同时检索，并将所有的检索结果进行聚合。多路检索的配置与单个检索器基本相同，只需额外开启`use_multi_retriever`，并将单个检索器的配置写入`multi_retriever_setting`下即可。

> [!NOTE]
> 打开多路检索后，前面单个检索器的设置会自动失效，检索相关的设置会从`multi_retriever_setting`中读取。


涉及的参数如下:

- **use_multi_retriever**  
  是否使用多个检索器。如果为 `True`，则可以设置多个检索器共同工作，进行多路检索。

- **multi_retriever_setting**  
  配置多个检索器的参数。通过该设置，您可以定义不同的检索方法和合并策略。

  - **merge_method**  
    设置合并检索结果的方式。支持三种方法：`concat`（拼接）、`rrf`（使用rrf算法合并）和 `rerank`（基于reranker进行重新排序）。

  - **topk**  
    设置最终保留的文档数量，仅在 `rrf` 和 `rerank` 合并方法中使用。

  - **rerank_model_name**  
    设置多路检索后使用的reranker模型名称，仅用于`merge_method`为`rerank`的情况。

  - **rerank_model_path**  
    设置重新排序模型的路径，仅用于`merge_method`为`rerank`的情况。

  - **retriever_list**  
    定义多个检索器的配置列表。每个检索器的参数设置与前面单个检索器的设置**完全相同**，可设置的参数包括但不仅限于：
      - `retrieval_method`: 检索方法（如 `e5` 或 `bm25`）。
      - `retrieval_topk`: 每个检索器返回的文档数量。
      - `index_path`: 检索索引的路径。
      - `retrieval_model_path`: 检索模型的路径。

    示例：
    ```yaml
    retriever_list:
      - retrieval_method: "e5"
        retrieval_topk: 5
        index_path: ~
        retrieval_model_path: ~
      - retrieval_method: "bm25"
        retrieval_topk: 5
        index_path: ~
        retrieval_model_path: ~
    ```
