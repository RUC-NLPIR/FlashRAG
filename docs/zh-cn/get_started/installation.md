# 安装FlashRAG

FlashRAG主体支持从PyPi和源码两种安装方式，需要Python版本>=3.10。默认安装的方式涵盖了FlashRAG本体以及所需要的大部分依赖，用户可以直接使用。由于部分依赖可能存在版本冲突，我们不设置自动安装，用户可以参考下文进行手动安装。

## 安装FlashRAG本体

### 从PyPi安装

> [!NOTE]
> 由于PyPI是夜间自动构建的，有时候可能会存在更新延迟的问题。如果需要使用最新的开发版本，建议使用源码安装。

直接使用pip从PyPi安装FlashRAG，需要使用`--pre`选项来安装最新的开发版本。

```bash
pip install flashrag-dev --pre
```

### 从源码安装

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

## 安装其他依赖

如果想要安装FlashRAG的所有依赖，可以使用:

```bash
# 安装全部依赖
pip install flashrag-dev[full] --pre
```

如果只想要安装某些部分，可以使用:

```bash
# 安装vllm来提升模型运行速度 (如果出现版本冲突，请基于vllm官方指南进行调整)
pip install vllm>=0.5.5

# 安装pyserini (用于BM25), 需要安装Java
pip install pyserini

# 安装sentence-transformers (用于Sentence-BERT)
pip install sentence-transformers

# 安装多模态相关依赖 (需要使用多模态RAG)
pip install flashrag-dev[multimodal] --pre
```

### 安装Faiss

Faiss是用于储存稠密索引的库，由于直接使用pip安装Faiss可能会出现驱动不匹配的问题，我们推荐使用conda进行安装。

> [!TIP]
> 如果需要使用DenseRetriever，则需要安装Faiss。
> 大部分场景只需要安装CPU版本，如果需要使用GPU，则需要安装GPU版本。

> [!TIP]
> 可能存在Faiss版本不匹配的问题，如果出现版本不匹配的问题，请基于Faiss官方指南进行调整。

```bash
# 只安装cpu版本
conda install -c pytorch faiss-cpu

# 安装gpu(+cpu)版本
conda install -c pytorch -c nvidia faiss-gpu
```

> [!NOTE]
> - 仅支持CPU的faiss-cpu conda包目前可在Linux（x86_64和arm64）、OSX（仅arm64）和Windows（x86_64）上使用
> - faiss-gpu，包含CPU和GPU索引，仅在Linux（x86_64）上支持CUDA 11.4和12.1



