# 欢迎来到 FlashRAG 的文档！

![](./asset/framework.jpg)

# 介绍

FlashRAG 是一个用于重现和开发检索增强生成（RAG）研究的 Python 工具包。我们的工具包包括 36 个预处理的基准 RAG 数据集和 16 种最先进的 RAG 算法。我们的特点如下：

- **广泛且可定制的框架**：包括 RAG 场景所需的基本组件，如检索器、重排序器、生成器和压缩器，允许灵活组装复杂的管道。

- **全面的基准数据集**：收集了 36 个预处理的 RAG 基准数据集，以测试和验证 RAG 模型的性能。

- **预实现的先进 RAG 算法**：提供 16 种先进的 RAG 算法及其报告结果，基于我们的框架。可以在不同设置下轻松重现结果。

- **高效的预处理阶段**：通过提供各种脚本（如用于检索的语料库处理、检索索引构建和文档的预检索）简化 RAG 工作流准备。

- **优化的执行**：该库的效率通过 vLLM、FastChat 等工具加速 LLM 推理，以及 Faiss 进行向量索引管理而得到提升。

- **易于使用的用户界面**：我们开发了一个非常易于使用的用户界面，可以轻松快速地配置和体验我们实现的 RAG 基线，并在可视化界面上运行评估脚本。

# 文档

- [安装](docs/installation.md)
- [功能](docs/features.md)
- [快速入门](docs/quick-start.md)
- [组件](docs/components.md)
- [FlashRAG-UI](docs/flashrag-ui.md)
- [支持的方法](docs/supporting-methods.md)
- [支持的数据集](docs/supporting-datasets.md)
- [常见问题](docs/faqs.md)

# :bookmark: 许可证

FlashRAG 根据 [<u>MIT 许可证</u>](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE) 进行许可。

# :star2: 引用

如果我们的论文对您的研究有帮助，请务必引用：

```BibTex
@article{FlashRAG,
  author       = {Jiajie Jin and
                  Yutao Zhu and
                  Xinyu Yang and
                  Chenghao Zhang and
                  Zhicheng Dou},
  title        = {FlashRAG: {A} Modular Toolkit for Efficient Retrieval-Augmented Generation
                  Research},
  journal      = {CoRR},
  volume       = {abs/2405.13576},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2405.13576},
  doi          = {10.48550/ARXIV.2405.13576},
  eprinttype    = {arXiv},
  eprint       = {2405.13576},
  timestamp    = {Tue, 18 Jun 2024 09:26:37 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2405-13576.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
