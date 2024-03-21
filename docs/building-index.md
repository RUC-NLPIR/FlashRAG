To build an index, you first need to save your corpus in `jsonl` format as follows, each line is a document.

```jsonl
{"id": "0", "contents": "contents for building index"}
{"id": "1", "contents": "contents for building index"}
```

We also support where document item contains both title and text:
```jsonl
{"id": "0", "title": "doc title", "text": "doc text"}
```
In this case, `contents` field will be set to ```"{title}"\n{text}``` for building index.



Then, use the following code to build your own index.

```bash
python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path /model/e5-base-v2/ \
    --corpus_path indexes/sample_corpus.jsonl \
    --save_dir indexes/ \
    --use_fp16 \
    --pooling_method 'mean'
```

* ```--pooling_method```: Due to the different pooling methods used by different embedding models, we may not have fully implemented them. To ensure accuracy, you can specify the pooling method corresponding to the retrieval model you are using (`mean`, `pooler` or `cls`). Except for some common embedding models (`e5`, `bge`, `dpr`), the default setting is `pooler`.



If building a bm25 index, there is no need to specify `model_path`:
```bash
python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path indexes/sample_corpus.jsonl \
    --save_dir indexes/ 
```


For **sparse retrieval method (BM25)**, we construct corpus as Lucene inverted indexes based on `Pyserini`.

For **dense retrieval methods**, especially the popular embedding models, we use `faiss` to build index and use `sqlite` for corpus storage to ensure extremely fast retrieval speed. After the construction is completed, the original corpus (in `jsonl` format) is no longer needed. We only need a sqlite database file and a faiss index file for subsequent retrieval.

