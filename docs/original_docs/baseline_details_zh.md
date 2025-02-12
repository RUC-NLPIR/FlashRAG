# 实验基准测试的实现细节

我们在统一的设置下测试了所有实现的方法。用户只需要下载相应的模型并填写配置，就可以使用我们提供的脚本 [这里](https://github.com/RUC-NLPIR/FlashRAG/blob/main/examples/methods/run_exp.py) 运行相应的结果。本章详细介绍了使用我们的工具包复现各种算法的具体实现细节，使用户能够轻松复制实验并将结果与我们的对齐。

## 全局设置

**检索器设置**：在我们的主要实验中，我们使用 [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2) 作为检索器，每个查询检索五个文档。我们使用2018年12月版的维基百科数据集的DPR版本作为我们的检索语料库，可以从我们的[数据集托管页面](https://huggingface.co/datasets/ignore/FlashRAG_datasets)下载。
在随后的检索实验中，我们同时使用 BM25 和 [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) 作为额外的检索器。BM25实验是使用 Pyserini 进行的。在索引构建期间，最大填充长度设置为512，而在检索期间最大查询填充长度设置为128。检索的批量大小为1024，启用了fp16。我们在索引中使用 Faiss Flat 索引以确保准确性。

**生成器设置**：我们在我们的主要实验中使用 [Llama3-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 作为生成器，最大输入长度为2048，最大输出长度为32。在生成过程中使用 vllm 框架进行推理，不进行采样。在随后的生成器实验中，我们使用 Qwen1.5-14B 作为额外的生成器。实验是使用四个 NVIDIA A100 80G GPU 进行的。

**提示设置**：我们使用统一的提示以确保公平性。具体来说，我们的系统提示是：

```
根据给定的文档回答问题。只给我答案，不要输出任何其他词。以下是给定的文档:{检索文档}
```
检索文档如下所列：
```
文档 1（标题:{标题}) {内容} 
文档 2（标题:{标题}) {内容}
```
我们的用户提示是：
```
问题:{问题}
```
这些提示是使用 `tokenizer.apply_chat_template` 函数组合的，作为最终输入到生成器模型。

**数据集设置**：对于有测试集的数据集，我们使用测试集进行测试。否则，我们使用开发集进行测试（即 nq, tqa, popqa, webq 是测试，其余是开发）。所有结果都使用了每个数据集的前1000个样本（在配置中将 `test_sample_num` 设置为1000，并关闭 `random_sample`）。我们已经在完整数据集上进行了测试，发现结果与1000个数据点的结果略有不同。

## 方法特定设置

除了上述的一般设置之外，每种方法通常都有自己的设置和配置。对于具有特定设置的方法，我们将在下面介绍每种方法自己的设置。

**AAR**：这项工作专注于优化检索器。对于我们的实验，我们使用了作者提供的预训练检索器 [这里](https://huggingface.co/OpenMatch/AAR-Contriever-KILT)。我们使用 AAR-Contriever-KILT 进行我们的结果，并支持 AAR-ANCE 实现。

**LLMLingua**：在这种方法中，我们使用 Llama2-7B 计算困惑度，并将 LongLLMLingua 作为压缩器，压缩率设置为0.55。其他参数设置为默认值。与原始的 LongLLMLingua 示例不同，我们使用检索到的文本作为输入到细化器（而不是整个提示）。我们的测试显示，Llama3 提示需要特殊 Token；使用原始设置会导致这些 Token 被省略，从而降低性能。

**RECOMP**：我们使用 RECOMP 提供的摘要模型进行我们的实验 [这里](https://huggingface.co/fangyuan)。对于 NQ、TQA 和 HotpotQA，我们使用相应的模型。对于剩余的三个数据集，没有可用的训练检查点。因此，我们使用 HotpotQA 检查点进行 2WikiMultihopQA，并使用 NQ 检查点进行 PopQA 和 WebQuestions。细化器的最大输入长度设置为1024，最大输出长度设置为512。

**Selective-Context**：我们使用 GPT2 输出困惑度，并将压缩率设置为0.5。与 LongLLMLingua 类似，我们使用检索到的文档作为输入到细化器。

**Ret-Robust**：这种方法专注于优化生成模型，使用 Self-Ask 提示方法进行训练。作者提供了在 NQ 和 2WikiMultihopQA 上训练的 LoRA 模型 [这里](https://huggingface.co/Ori/llama-2-13b-peft-nq-retrobust)。因此，我们使用加载了相应 LoRA 的 Llama2-13B 模型进行测试。由于 HotpotQA 没有训练模型，我们使用 2WikiMultihopQA LoRA。对于剩余的数据集，我们使用 NQ LoRA。我们将最大交互轮次设置为5，最大输出 Token 设置为100。对于 HotpotQA 和 2WikiMultihopQA，我们禁用了 `single_hop` 设置，以允许过程自动将复杂查询分解为多个迭代。

**SuRe**：这种方法提示模型生成候选答案、评分并对它们进行排名，选择最佳答案。为确保一致性，我们使用原始论文中提供的提示，这些提示可以与我们的代码实现一起参考。

**SKR**：我们实现了 SKR-knn 方法，它需要一个编码器模型和推理时的训练数据。具体来说，它根据输入查询从训练数据中识别最相似的查询，确定输入查询是否需要检索。我们的库包括作者提供的训练数据；相应的编码器模型可以从 [这里](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased) 下载。

**Self-RAG**：我们使用 Self-RAG 提供的 Llama2-7B 检查点 [这里](https://huggingface.co/selfrag/selfrag_llama2_7b)，将最大输出 Token 设置为100以确保正常运行。温度设置为0，`top_p` 设置为1。

**IRCoT**：对于所有实验，我们使用一次示例来添加提示。示例来自 IRCoT 提供的 [演示文件](https://github.com/StonyBrookNLP/ircot/blob/main/prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt)。Max iter 设置为2。

**Trace**：这种方法需要首先从搜索结果中提取三元组，然后构建推理链。这两个步骤依赖于 LLM 的提示。遵循原始工作，我们使用 Llama3-8B-instruct 来执行这些步骤，在每个提示中使用3个示例。对于没有示例的数据集，我们使用 2WikiMultihopQA 的示例作为替代。其他超参数遵循我们代码中的默认设置。

**Spring**：这个模型需要将虚拟 Token 的嵌入纳入训练，除了它自己的生成器之外。由于只训练了 llama2 系列的模型，我们在 `llama2-7B-chat` 上进行了实验。

**Adaptive-RAG**: 这个方法需要一个分类器对查询进行分类。由于作者没有提供官方的checkpoint模型，我们在实验中使用了其他人在Huggingface上训练的模型（这可能会导致结果不一致）。如果官方开源模型在未来发布，我们将更新实验结果。
