# Implementation Details for Benchmarking Experiments

We tested all implemented methods under a unified setting. Users only need to download the corresponding model and fill in the config to run the corresponding results using the script we provide [here](https://github.com/RUC-NLPIR/FlashRAG/blob/main/examples/methods/run_exp.py). This chapter details the implementation specifics of reproducing various algorithms using our toolkit, allowing users to effortlessly replicate the experiments and align their results with ours.

## Global Setting

**Retriever Setting**: In our main experiments, we utilize [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2) as the retriever, retrieving five documents per query. We use the DPR version of the Wikipedia December 2018 dataset as our retrieval corpus, which can be downloaded from our dataset hosting page [here](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets). 
In subsequent retrieval experiments, we employ both BM25 and [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) as additional retrievers. The BM25 experiments are conducted using Pyserini. During index construction, the maximum padding length is set to 512, while the maximum query padding length is set to 128 during retrieval. The batch size for retrieval is 1024, with fp16 enabled. We employ the Faiss Flat index in indexing for accuracy.

**Generator Setting**: We employ [Llama3-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) as the generator in our main experiment, with a maximum input length of 2048 and a maximum output length of 32. Inference is performed using the vllm framework without sampling during generation. In subsequent generator experiments, we employ Qwen1.5-14B as additional generators. The experiments are conducted using four NVIDIA A100 80G GPUs.

**Prompt Setting**: We use a unified prompt to ensure fairness. Specifically, our system prompt is:

```
Answer the question based on the given document. Only give me the answer and do not output any other words. The following are given documents:{retrieval documents}
```
The retrieval documents are listed as follows:
```
Doc 1 (Title:{title}) {content} 
Doc 2 (Title:{title}) {content}
```
Our user prompt is:
```
Question:{question}
```
These prompts are combined using the `tokenizer.apply_chat_template` function, serving as the final input to the generator model.

**Dataset Setting**: For datasets with a test set, we use the test set for testing. Otherwise, we use the dev set for testing (i.e. nq, tqa, popqa, webq is test, and the rest is dev). All results used the first 1000 samples from each dataset (set `test_sample_num` to 1000 in config and turn off `random_sample`). We have tested on a full dataset and found that the results differ slightly from those of 1000 data points.

## Method Specific Setting

In addition to the general settings mentioned above, each method often has its own settings and configurations. For methods with specific settings, we will introduce each method's own settings below.

**AAR**: This work focuses on optimizing the retriever. For our experiments, we utilize the pre-trained retriever provided by the authors [here](https://huggingface.co/OpenMatch/AAR-Contriever-KILT). We use AAR-Contriever-KILT for our results and also support the AAR-ANCE implementation.

**LLMLingua**: In this method, we use Llama2-7B to compute perplexity and LongLLMLingua as the compressor, with a compression rate set to 0.55. Other parameters were set to default values. Unlike the original LongLLMLingua example, we use the retrieved text as input to the refiner (instead of the entire prompt). Our tests reveal that the Llama3 prompt requires special tokens; using the original setting caused these tokens to be omitted, resulting in degraded performance.

**RECOMP**: We use the abstractive model provided by RECOMP for our experiments [here](https://huggingface.co/fangyuan). For NQ, TQA, and HotpotQA, we use the corresponding models. For the remaining three datasets, there were no trained checkpoints available. Therefore, we use the HotpotQA checkpoint for 2WikiMultihopQA, and the NQ checkpoint for PopQA and WebQuestions. The maximum input length for the refiner is set to 1024, and the maximum output length to 512.

**Selective-Context**: We use GPT2 to output perplexity and set the compression rate to 0.5. Similar to LongLLMLingua, we use the retrieved documents as input to the refiner.

**Ret-Robust**: This method focuses on optimizing the generative model, training it using the Self-Ask prompt method. The authors provided LoRA models trained on NQ and 2WikiMultihopQA [here](https://huggingface.co/Ori/llama-2-13b-peft-nq-retrobust). Consequently, we test using the Llama2-13B model loaded with the corresponding LoRA. As there is no trained model for HotpotQA, we use the 2WikiMultihopQA LoRA. For the remaining datasets, we use the NQ LoRA. We set the maximum interaction rounds to 5 and the maximum output tokens to 100. For HotpotQA and 2WikiMultihopQA, we disable the `single_hop` setting to allow the process to automatically decompose complex queries into multiple iterations.

**SuRe**: This method prompts the model to generate candidate answers, scores, and ranks them, selecting the best one. To ensure consistency, we use the prompts provided in the original paper, which can be referenced alongside our code implementation.

**SKR**: We implement the SKR-knn method, which requires an encoder model and inference-time training data. Specifically, it identifies the most similar queries from the training data based on the input query, determining whether the input query needs retrieval. Our library includes the training data provided by the authors; the corresponding encoder model can be downloaded [here](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased).

**Self-RAG**: We use the Llama2-7B checkpoint provided by Self-RAG [here](https://huggingface.co/selfrag/selfrag_llama2_7b), setting the max output tokens to 100 to ensure proper operation. The temperature is set to 0, and `top_p` is set to 1.

**IRCoT**: For all experiments, we used one shot example to add prompts. The example comes from [the demonstration file](https://github.com/StonyBrookNLP/ircot/blob/main/prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt) provided by IRCoT. Max iter is set to 2.

**Trace**: This method requires first extracting triples from the search results and then constructing a reasoning chain. These two steps depend on the prompt of the feed shot for LLM. Follow the original work, we use Llama3-8B-instruct to do these steps, use 3 examplars in each prompt. For datasets that don't have examplars, we use the examplars from 2WikiMultihopQA as a substitute. Other hyperparameters follow default settings in our code.

**Spring**: This model needs to incorporate the embedding of virtual tokens for training on top of its own generator. Due to only training models from the llama2 series, we conducted experiments on `llama2-7B-chat`.


**Adaptive-RAG**: This method requires a classifier to classify the query. Since the author did not provide an official checkpoint, we used a checkpoint trained by others on Huggingface for the experiment (which may result in inconsistent results). If the official open-source checkpoint is released in the future, we will update the experimental results.