# 索引构建

基于准备好的检索文档库，可以对需要使用的检索方法构建索引。FlashRAG目前支持纯文本模态的检索(embedding & BM25)以及多模态检索方法(clip-based)。

## 纯文本模态

* 对于**密集检索方法**，特别是常用的嵌入模型 (e5, bge等)，我们使用 `faiss` 来构建索引。

* 对于**稀疏检索方法(BM25)**，我们基于 `Pyserini` 或 `bm25s` 将语料库构建为 Lucene 倒排索引。构建的索引包含原始文档。

这两种方法对应的索引构建启动脚本为同一个脚本，只需设置不同的参数即可。

> [!TIP]
> 对于多模态文档库，支持对每条数据的文本模态使用纯文本模态的索引构建方式。只需直接将多模态的文档库作为`corpus_path`输入即可自动处理。

### 稠密检索方法

FlashRAG支持使用各类embedding模型作为稠密检索方法(如E5,BGE,DPR等)，修改以下代码中的参数为您自己的参数。

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

* `--pooling_method`：如果未指定，我们将根据模型名称和模型文件自动选择。但由于不同的嵌入模型使用不同的池化方法，**我们可能没有完全实现它们**。为确保准确性，您可以**指定与您使用的检索模型相对应的池化方法**（`mean`、`pooler` 或 `cls`）。

* `--instruction`：某些嵌入模型在编码前需要额外的指令来连接查询，可以在这里指定。目前，我们会自动为 **E5** 和 **BGE** 模型填充指令，而其他模型则需要手动补充。

如果检索模型支持 `sentence transformers` 库，您可以使用以下代码来构建索引（**无需考虑池化方法**）：

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

### 稀疏检索方法 (BM25)

如果构建 BM25 索引，则无需指定 `model_path`。目前我们支持使用两种引擎进行BM25的索引构建: `BM25s`和`Pyserini`，选择任一即可。`Pyserini`的安装较麻烦，需要同时安装`java`作为依赖，但整体稳定性较好，对中文的支持较好，推荐使用。`BM25s`安装简单，不包含复杂依赖，检索效果与`Pyserini`相似，适合在不方便安装Pyserini时使用。

> [!NOTE]
> 构建索引使用的引擎和检索时使用的引擎需保持一致，否则会出现索引文件不匹配的问题。


#### 使用 BM25s 构建索引

```bash
python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path indexes/sample_corpus.jsonl \
    --bm25_backend bm25s \
    --save_dir indexes/ 
```

#### 使用 Pyserini 构建索引

```bash
python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path indexes/sample_corpus.jsonl \
    --bm25_backend pyserini \
    --save_dir indexes/ 
```

## 多模态检索

目前FlashRAG支持使用Clip系列的模型做图文模态的混合检索，即使用单个Clip模型对文本和图片生成embedding。

![](../asset/CLIP.png)

上节得到的多模态corpus为下面的格式，每条数据同时包含了文本和图片，因此在构建索引时会对两种数据分别得到embedding(也可设置只做某个模态)。

```python
{
    'id': str,
    'text': str,
    'image': str # 图片对应的字节流数据
}
```

构建多模态索引的脚本如下:

```bash
python -m flashrag.retriever.index_builder \
    --retrieval_method jina-clip-v2 \
    --model_path model/jina-clip-v2 \
    --corpus_path datasets/mathvista/train.parquet \
    --save_dir indexes/mathvista \
    --max_length 512 \
    --batch_size 512 \
    --faiss_type Flat \
    --index_modal all
```

* **index_modal**：指定需要构建索引的模态，可以设置为: 'text', 'image'或'all'. 每个指定的模块都会生成一个对应的index文件。

> [!TIP]
> 如果需要对文本模态使用纯文本模态的模型(比如BM25, Embedding等)，可以直接使用上面纯文本模态的索引构建方法进行构建。只需直接替换`corpus_path`为多模态我文档即可，`text`会被自动设置为待索引文本。




