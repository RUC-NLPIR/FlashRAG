# Welcome to FlashRAG's documentation!

![](./asset/framework.jpg)

# Introduction

FlashRAG is a Python toolkit for the reproduction and development of Retrieval Augmented Generation (RAG) research. Our toolkit includes 36 pre-processed benchmark RAG datasets and 16 state-of-the-art RAG algorithms. Our features are as follows:

- **Extensive and Customizable Framework**: Includes essential components for RAG scenarios such as retrievers, rerankers, generators, and compressors, allowing for flexible assembly of complex pipelines.

- **Comprehensive Benchmark Datasets**: A collection of 36 pre-processed RAG benchmark datasets to test and validate RAG models' performances.

- **Pre-implemented Advanced RAG Algorithms**: Features 16 advancing RAG algorithms with reported results, based on our framework. Easily reproducing results under different settings.

- **Efficient Preprocessing Stage**: Simplifies the RAG workflow preparation by providing various scripts like corpus processing for retrieval, retrieval index building, and pre-retrieval of documents.

- **Optimized Execution**: The library's efficiency is enhanced with tools like vLLM, FastChat for LLM inference acceleration, and Faiss for vector index management.

- **Easy to Use UI** : We have developed a very easy to use UI to easily and quickly configure and experience the RAG baselines we have implemented, as well as run evaluation scripts on a visual interface.


# Documentation

- [Installation](docs/installation.md)
- [Features](docs/features.md)
- [Quick-Start](docs/quick-start.md)
- [Components](docs/components.md)
- [FlashRAG-UI](docs/flashrag-ui.md)
- [Supporting Methods](docs/supporting-methods.md)
- [Supporting Datasets](docs/supporting-datasets.md)
- [FAQs](docs/faqs.md)

# :bookmark: License

FlashRAG is licensed under the [<u>MIT License</u>](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE).

# :star2: Citation

Please kindly cite our paper if helps your research:

```BibTex
@article{FlashRAG,
    author={Jiajie Jin and
            Yutao Zhu and
            Xinyu Yang and
            Chenghao Zhang and
            Zhicheng Dou},
    title={FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research},
    journal={CoRR},
    volume={abs/2405.13576},
    year={2024},
    url={https://arxiv.org/abs/2405.13576},
    eprinttype={arXiv},
    eprint={2405.13576}
}
```