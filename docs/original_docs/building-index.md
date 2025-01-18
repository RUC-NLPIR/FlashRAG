## Indexing your own corpus


### Step1: Prepare corpus
To build an index, you first need to save your corpus in `jsonl` format as follows, each line is a document.  

```jsonl
{"id": "0", "contents": "contents for building index"}
{"id": "1", "contents": "contents for building index"}
```

If you want to use Wikipedia as a corpus, you can refer to our documentation for [process Wikipedia](./process-wiki.md) to convert it into an indexed format.


### Step2: Indexing

Then, use the following code to build your own index.


* For **dense retrieval methods**, especially the popular embedding models, we use `faiss` to build index. 

* For **sparse retrieval method (BM25)**, we construct corpus as Lucene inverted indexes based on `Pyserini` or `bm25s`. The constructed index contains the original doc.


#### For dense retrieval methods

Modify the parameters in the following code to yours.

```bash
python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path /model/e5-base-v2/ \
    --corpus_path indexes/sample_corpus.jsonl \
    --save_dir indexes/ \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --pooling_method mean \
    --faiss_type Flat 
```

* ```--pooling_method```: If this is not specified, we will automatically select based on the model name and model file. However, due to the different pooling methods used by different embedding models, **we may not have fully implemented them**. To ensure accuracy, you can **specify the pooling method corresponding to the retrieval model** you are using (`mean`, `pooler` or `cls`).

* ```---instruction```: Some embedding models require additional instructions to concatenate the query before encoding, which can be specified here. At present, we will automatically fill in the instructions for **E5** and **BGE** models, while other models need to be manually supplemented.

If the retrieval model support `sentence transformers` library, you can use following code to build index (**no need to consider pooling method**).

```bash
python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path /model/e5-base-v2/ \
    --corpus_path indexes/sample_corpus.jsonl \
    --save_dir indexes/ \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --pooling_method mean \
    --sentence_transformer \
    --faiss_type Flat 
```




#### For sparse retrieval method (BM25)

If building a bm25 index, there is no need to specify `model_path`.

##### Use BM25s to build index

```bash
python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path indexes/sample_corpus.jsonl \
    --bm25_backend bm25s \
    --save_dir indexes/ 
```

##### Use Pyserini to build index

```bash
python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path indexes/sample_corpus.jsonl \
    --bm25_backend pyserini \
    --save_dir indexes/ 
```




