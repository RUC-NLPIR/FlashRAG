# RAG-Components Usage

## Retriever

The retriever takes a series of queries and retrieves the top k documents from a corpus corresponding to each query.

Load the retriever with `flashrag.utils.get_retriever`. It uses `retrieval_method` to select the retriever and loads any corpus, index, or model that retriever needs.

```python
retriever = get_retriever(config)
```

Then, you can use `retriever.search` (for a single query) or `retriever.batch_search` (for a list of queries) to perform retrieval.

```python
# if you set `return_score=True`
retrieval_results, scores = retriever.batch_search(input_query_list, return_score=True)

# if you set `return_score=False`
retrieval_results = retriever.batch_search(input_query_list, return_score=False)
```

When using `batch_search`, `retrieval_results` is a two-level nested list like:

```python
[
    [doc1, doc2, ...],  # doc items for query1
    [doc1, doc2, ....],  # doc items for query2
    ...
]
```

When using `search`, `retrieval_results` is a regular list containing the top k doc item for each query.

Each doc item is a dictionary containing `doc_id`, `title`, `contents`, etc. (similar to the contents in the corpus).

`scores` have the same format as `retrieval_results`, except that each `doc_item` is replaced by a `float` value representing the matching score between the document and the query provided by the retriever.

#### Additional features of the Retriever

The `search` and `batch_search` functions of the retriever implement three additional functionalities using two decorator functions:
- **Pre-load retrieval results**: Suitable for cases where you provide some retrieval results for queries yourself. If `use_retrieval_cache` is set in the config and a cache file is provided, it first checks whether the cache file contains the retrieval results for the corresponding queries and returns them if available.
- **Save retrieval results**: If `save_retrieval_cache` is set to True, the retriever will save the retrieval results for each query as a cache, making it easy to use the cache directly next time.
- **Rerank**: If `use_reranker=True`, the `search` function will integrate reranking to further sort the retrieval results.

### Parallel web retrieval

Use Python 3.10 or newer and a Faiss build that supports NumPy 2. From your FlashRAG checkout, create a fresh environment and install the Parallel extra and the standard pipeline's `termcolor` dependency. This example uses Python 3.12 with CPU Faiss. See the [Faiss installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for platform support and GPU builds.

```bash
conda create -n flashrag-parallel -c pytorch -c conda-forge python=3.12 pip faiss-cpu=1.16.0
conda activate flashrag-parallel
python -m pip install -e '.[parallel]' termcolor
```

Add these settings to your existing pipeline configuration.

```yaml
retrieval_method: parallel
use_multi_retriever: false
retrieval_topk: 5
```

`SequentialPipeline(config)` selects Parallel through the retriever factory. It sends uncached text queries to `https://mcp.parallel.ai/mcp` and uses returned excerpts as context. Parallel retrieval works without a local index, corpus, or API key. The pipeline uses the generator configured for your experiment.

Each result includes the source URL, title, excerpt text, and `contents` in FlashRAG's title-then-text format. The URL also serves as the document ID. `retrieval_topk` caps the number of results locally. Before any configured reranking, scores are the reciprocal ranks of the returned documents.

## Generator


The Generator takes prompts as input and returns outputs corresponding to each prompt.

You can load the Generator using `flashrag.utils.get_generator`. Depending on the name of the input generator model, it will choose to load a different structure of the generator model.

```python
generator = get_generator(config)
```

Then, use the `generate` method for generation.

```python
input_list = ['who is taylor swift?', 'who is jack ma?']
result = generator.generate(input_list)
```

You can obtain the generation probability of each token by using `return_scores`.

```python
result, scores = generator.generate(input_list, return_scores=True)
```

The `generate` function can also accept parameters needed for generation, such as `topk`, `do_sample`, etc. These parameters can also be specified in the config, but the ones specified in `generate` take precedence.

```python
result = generator.generate(input_list, top_p=1.0, max_tokens=32)
```

## Config

`Config` class supports using `.yaml` files as input or variables as input. The priority of variables is higher than that of files. **All subsequent component settings depend on `Config`.**

If there are variables that need to be used that are not specified in these two places, default values will be loaded (`basic_config.yaml`).

```python
from flashrag.config import Config

config_dict = {"retrieval_method": "bge"}
config = Config('my_config.yaml', config_dict = config_dict)
print(config['retrieval_method'])
```


