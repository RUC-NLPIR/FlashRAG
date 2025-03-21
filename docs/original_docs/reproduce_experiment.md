# Guidelines for Reproduction Methods

In this document, we will introduce how to reproduce the results of various methods listed in our table under a unified setting. For specific settings and explanations of each method, please refer to [implementation details](./baseline_details.md). It is recommended to have some basic understanding of our repository beforehand, which can be found in [introduction for beginners](./introduction_for_beginners_en.md).

## Preliminary

- Install FlashRAG and dependencies
- Download [Llama3-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
- Download datasets (you can download from our repo: [here](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets))
- Download retrieval corpus (from [here](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus))
- Build index for retrieval, using E5 for embedding (see [how to build index?](./building-index.md))

## Reproduce Step

All the code used is based on the repository's [example/methods](../examples/methods/). We have set appropriate hyperparameters for various methods. If you need to adjust them yourself, you can refer to the config dictionary provided for each method and the original papers of each method.

### 1. Set Basic Config

First, you need to fill in the paths of various downloads in `my_config.yaml`. Specifically, you need to fill in the following four fields:
- **model2path**: Replace the paths of E5 and Llama3-8B-instruct models with your own paths
- **method2index**: Fill in the path of the index file built using E5
- **corpus_path**: Fill in the path of the Wikipedia corpus file in `jsonl` format
- **data_dir**: Change to the download path of your own dataset

### 2. Set Config for Specific Method

For some methods that require the use of additional models, extra steps are required. We will introduce the methods that need extra steps below. If you know that the method you want to run does not need these steps, you can skip directly to the third section.

Table of Contents:
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

This method requires using a new retriever, so you need to download the retriever and build the index.

- Additional Step1: Download AAR-Contriever (from [here](https://huggingface.co/OpenMatch/AAR-Contriever-KILT))
- Additional Step2: Build the index for AAR-Contriever (note that the pooling method should be 'mean')
- Additional Step3: Modify the `index_path` and `model2path` in the `AAR` function in `run_exp.py`.

#### LongLLMLingua

This method requires downloading Llama2-7B.

- Additional Step1: Download Llama2-7B (from [here](https://huggingface.co/meta-llama/Llama-2-7b-hf))
- Additional Step2: Modify the `refiner_model_path` in the `llmlingua` function in `run_exp.py`

#### RECOMP

This method requires downloading three checkpoints trained by the authors (trained on NQ, TQA, and HotpotQA respectively).

- Additional Step1: Download the author's checkpoints ([NQ Model](https://huggingface.co/fangyuan/nq_abstractive_compressor), [TQA Model](https://huggingface.co/fangyuan/tqa_abstractive_compressor), [HotpotQA Model](https://huggingface.co/fangyuan/hotpotqa_abstractive))
- Additional Step2: Fill in the downloaded model paths in the `model_dict` of the `recomp` function

#### Selective-Context

This method requires downloading GPT2.

- Additional Step1: Download GPT2 (from [here](https://huggingface.co/openai-community/gpt2))
- Additional Step2: Modify the `refiner_model_path` in the `sc` function in `run_exp.py`

#### Ret-Robust

This method requires downloading the Lora trained by the authors and downloading the Llama2-13B model to load the Lora.

- Additional Step1: Download Llama2-13B (from [here](https://huggingface.co/meta-llama/Llama-2-13b-hf))
- Additional Step2: Download the author's trained Lora, trained on NQ (from [here](https://huggingface.co/Ori/llama-2-13b-peft-nq-retrobust)) and trained on 2WikiMultihopQA (from [here](https://huggingface.co/Ori/llama-2-13b-peft-2wikihop-retrobust))
- Additional Step3: Modify the corresponding Lora paths in the `model_dict` of the `retrobust` function and the Llama2-13B path in `my_config.yaml`

We recommend adjusting the `single_hop` parameter in the `SelfAskPipeline` according to different datasets, which controls whether to decompose the query. For `NQ, TQA, PopQA, WebQ`, we set `single_hop` to `True`.

#### SKR

This method requires an embedding model and training data used during the inference stage. We provide the training data given by the authors. If you wish to use your own training data, you can generate it according to the format of the training data and the original paper.

- Additional Step1: Download the embedding model (from [here](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased))
- Additional Step2: Download the training data (from [here](../examples/methods/sample_data/skr_training.json))
- Additional Step3: Fill in the embedding model path in the `model_path` of the `skr` function
- Additional Step4: Fill in the training data path in the `training_data_path` of the `skr` function

#### Self-RAG

This method requires using a trained model and currently only supports running in the `vllm` framework.

- Additional Step1: Download the Self-RAG model (from [7B model](https://huggingface.co/selfrag/selfrag_llama2_7b), [13B model](https://huggingface.co/selfrag/selfrag_llama2_13b))
- Additional Step2: Modify the `generator_model_path` in the `selfrag` function.

#### Spring
This method requires a virtual token embedding file and currently only supports running in the `hf` framework.

- Additional Step1: Download virtual token embedding file from [official repo](https://huggingface.co/yutaozhu94/SPRING)
- Additional Step2: Modify the `token_embedding_path` in the `spring` function.

#### Adaptive-RAG

This method requires a classifier to classify the query. Since the author did not provide an official checkpoint, we used a checkpoint trained by others on Huggingface for the experiment (which may result in inconsistent results).

If the official open-source checkpoint is released in the future, we will update the experimental results.

- Additional Step1: Download classifier model from huggingface repo (**not official**): [illuminoplanet/combined_flan_t5_xl_classifier](https://huggingface.co/illuminoplanet/combined_flan_t5_xl_classifier)
- Additional Step2: Modify the `model_path` in `adaptive` function.

#### RQRAG

This method requires downloading the RQRAG model.

- Additional Step1: Download RQRAG model from huggingface repo: [zorowin123/rq_rag_llama2_7B](https://huggingface.co/zorowin123/rq_rag_llama2_7B)
- Additional Step2: Modify the `generator_model_path` in the `rqrag` function.

### 3. Run methods

Run the experiment on the NQ dataset using the following command.

```bash
python run_exp.py --method_name 'naive' \
                  --split 'test' \
                  --dataset_name 'nq' \
                  --gpu_id '0,1,2,3'
```

The method can be selected from the following:
```
naive zero-shot AAR-contriever llmlingua recomp selective-context sure replug skr flare iterretgen ircot trace
```


#### R1-Searcher

This method requires downloading the R1-Searcher model.

- Additional Step1: Download R1-Searcher model from huggingface repo: [XXsongLALA/Qwen-2.5-7B-base-RAG-RL](https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL)
- Additional Step2: Modify the `generator_model_path` in the `r1searcher` function.


