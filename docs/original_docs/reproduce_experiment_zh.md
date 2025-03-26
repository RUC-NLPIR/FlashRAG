在本文档中，我们将介绍如何在统一的设置下复现我们表格中列出的各种方法的结果。有关每种方法的具体设置和解释，请参考[实现细节](./baseline_details.md)。建议事先对本仓库有一定的基本了解，可以在[初学者介绍](./introduction_for_beginners_en.md)中找到。

## 预备工作

- 安装 FlashRAG 及其依赖项
- 下载 [Llama3-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)，[E5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
- 下载数据集（你可以从我们的仓库下载：[这里](https://huggingface.co/datasets/ignore/FlashRAG_datasets))
- 下载检索语料库（从[这里](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)下载）
- 使用 E5 构建检索索引（参见[如何构建索引？](./building-index.md)）

## 复现步骤

所有使用的代码都基于仓库的 [example/methods](../examples/methods/)。我们为各种方法设置了适当的超参数。如果你需要自己调整它们，可以参考为每种方法提供的配置字典以及每种方法的原始论文。

### 1. 设置基本配置

首先，你需要在 `my_config.yaml` 中填写各种下载的路径。具体来说，你需要填写以下四个字段：
- **model2path**：用你自己的路径替换 E5 和 Llama3-8B-instruct 模型的路径
- **method2index**：填写使用 E5 构建的索引文件的路径
- **corpus_path**：填写 `jsonl` 格式的 Wikipedia 语料库文件的路径
- **data_dir**：更改为你自己的数据集下载路径

### 2. 设置特定方法的配置

对于一些需要使用额外模型的方法，需要额外的步骤。我们将在下面介绍需要额外步骤的方法。如果你知道你想要运行的方法不需要这些步骤，你可以直接跳到第三部分。

目录：
- [AAR](#aar)
- [LongLLMLingua](#longllmlingua)
- [RECOMP](#recomp)
- [Selective-Context](#selective-context)
- [Ret-Robust](#ret-robust)
- [SKR](#skr)
- [Self-RAG](#self-rag)
- [Spring](#spring)
- [Adaptive-RAG](#adaptive-rag)
- [RQRAG](#rqrag)
- [R1-Searcher](#r1-searcher)

#### AAR

这种方法需要使用一个新的检索器，因此你需要下载检索器并构建索引。

- 额外步骤1：下载 AAR-Contriever（从[这里](https://huggingface.co/OpenMatch/AAR-Contriever-KILT)）
- 额外步骤2：为 AAR-Contriever 构建索引（注意，池化方法应该是 'mean'）
- 额外步骤3：在 `run_exp.py` 中的 `AAR` 函数中修改 `index_path` 和 `model2path`

#### LongLLMLingua

这种方法需要下载 Llama2-7B。

- 额外步骤1：下载 Llama2-7B（从[这里](https://huggingface.co/meta-llama/Llama-2-7b-hf)）
- 额外步骤2：在 `run_exp.py` 中的 `llmlingua` 函数中修改 `refiner_model_path`

#### RECOMP

这种方法需要下载作者训练的三个检查点（分别在 NQ、TQA 和 HotpotQA 上训练）。

- 额外步骤1：下载作者的检查点（[NQ 模型](https://huggingface.co/fangyuan/nq_abstractive_compressor)，[TQA 模型](https://huggingface.co/fangyuan/tqa_abstractive_compressor)，[HotpotQA 模型](https://huggingface.co/fangyuan/hotpotqa_abstractive)）
- 额外步骤2：在 `recomp` 函数的 `model_dict` 中填写下载的模型路径

#### Selective-Context

这种方法需要下载 GPT2。

- 额外步骤1：下载 GPT2（从[这里](https://huggingface.co/openai-community/gpt2)）
- 额外步骤2：在 `run_exp.py` 中的 `sc` 函数中修改 `refiner_model_path`

#### Ret-Robust

这种方法需要下载作者训练的 Lora 并下载 Llama2-13B 模型以加载 Lora。

- 额外步骤1：下载 Llama2-13B（从[这里](https://huggingface.co/meta-llama/Llama-2-13b-hf)）
- 额外步骤2：下载作者训练的 Lora，训练于 NQ（从[这里](https://huggingface.co/Ori/llama-2-13b-peft-nq-retrobust)）和训练于 2WikiMultihopQA（从[这里](https://huggingface.co/Ori/llama-2-13b-peft-2wikihop-retrobust)）
- 额外步骤3：在 `retrobust` 函数的 `model_dict` 中修改相应的 Lora 路径，并在 `my_config.yaml` 中修改 Llama2-13B 路径

我们建议根据不同的数据集调整 `SelfAskPipeline` 中的 `single_hop` 参数，该参数控制是否分解查询。对于 `NQ, TQA, PopQA, WebQ`，我们将 `single_hop` 设置为 `True`。

#### SKR

这种方法需要一个嵌入模型和推理阶段使用的训练数据。我们提供了作者给定的训练数据。如果你想使用自己的训练数据，可以根据训练数据的格式和原始论文生成它。

- 额外步骤1：下载嵌入模型（从[这里](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased)）
- 额外步骤2：下载训练数据（从[这里](../examples/methods/sample_data/skr_training.json)）
- 额外步骤3：在 `skr` 函数的 `model_path` 中填写嵌入模型路径
- 额外步骤4：在 `skr` 函数的 `training_data_path` 中填写训练数据路径

#### Self-RAG

这种方法需要使用一个训练好的模型，目前只支持在 `vllm` 框架中运行。

- 额外步骤1：下载 Self-RAG 模型（从 [7B 模型](https://huggingface.co/selfrag/selfrag_llama2_7b)，[13B 模型](https://huggingface.co/selfrag/selfrag_llama2_13b)）
- 额外步骤2：在 `selfrag` 函数中修改 `generator_model_path`

#### Spring
这种方法需要一个虚拟 Token 嵌入文件，目前只支持在 `hf` 框架中运行。

- 额外步骤1：从[官方仓库](https://huggingface.co/yutaozhu94/SPRING)下载 embedding 文件
- 额外步骤2：在 `spring` 函数中修改 `token_embedding_path`

#### Adaptive-RAG

该方法需要一个分类器对查询进行分类。由于作者没有提供官方的检查点，我们使用了其他人在Huggingface上训练的检查点进行实验（这可能会导致结果不一致）。

后续如果官方开源了checkpoint，我们将更新实验结果。

- 额外步骤1：从 Huggingface 仓库下载分类器模型（**非官方**）：[illuminoplanet/combined_flan_t5_xl_classifier](https://huggingface.co/illuminoplanet/combined_flan_t5_xl_classifier)
- 额外步骤2：修改 `adaptive` 函数中的 `model_path`。

#### RQRAG

这种方法需要下载 RQRAG 模型。

- 额外步骤1：从 Huggingface 仓库下载 RQRAG 模型：[zorowin123/rq_rag_llama2_7B](https://huggingface.co/zorowin123/rq_rag_llama2_7B)
- 额外步骤2：在 `rqrag` 函数中修改 `generator_model_path`


### 3. 运行方法

使用以下命令在 NQ 数据集上运行实验。

```bash
python run_exp.py --method_name 'naive' \
                  --split 'test' \
                  --dataset_name 'nq' \
                  --gpu_id '0,1,2,3'
```

可以从以下方法中选择：
```
naive zero-shot AAR-contriever llmlingua recomp selective-context sure replug skr flare iterretgen ircot trace
```


#### R1-Searcher

这种方法需要下载 R1-Searcher 模型。

- 额外步骤1：从 Huggingface 仓库下载 R1-Searcher 模型：[XXsongLALA/Qwen-2.5-7B-base-RAG-RL](https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL)
- 额外步骤2：在 `r1searcher` 函数中修改 `generator_model_path`
