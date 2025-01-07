# Multi-Retriever Usage

Using multiple retrievers simultaneously requires configuration in the `multi_retrieve` section of the config file. Enable `use_multi_retriever` and configure it as follows:

```yaml
use_multi_retriever: True # whether to use multi retrievers
multi_retriever_setting:
  merge_method: "concat" # support 'concat', 'rrf', 'rerank'
  topk: 5 # final remaining documents, only used in 'rrf' and 'rerank' merge
  rerank_model_name: ~
  rerank_model_path: ~
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

The `retriever_list` can include multiple retrievers, each with its own corresponding corpus and index. The configuration for each retriever follows the same settings as a single retriever.

Currently, three aggregation methods are supported:

1. **`concat`**: Directly concatenate the results from multiple retrievers.
2. **`rrf`**: Aggregate the results from retrievers using the RRF (Reciprocal Rank Fusion) algorithm.
3. **`rerank`**: Use a reranker to rerank all the results from the retrievers. This requires configuring the reranker.

For both `rrf` and `rerank`, only the top `k` results will be retained.
