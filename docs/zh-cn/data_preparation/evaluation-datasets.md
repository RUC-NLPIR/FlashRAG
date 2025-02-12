# 数据集概况

FlashRAG预处理了38个RAG中常用的数据集，并托管在[Huggingface](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)和[ModelScope](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)双平台上。

## 数据集信息

目前FlashRAG已经处理好的数据集信息如下：

| Task                      | Dataset Name    | Knowledge Source | # Train   | # Dev   | # Test |
| ------------------------- | --------------- | ---------------- | --------- | ------- | ------ |
| QA                        | NQ              | wiki             | 79,168    | 8,757   | 3,610  |
| QA                        | TriviaQA        | wiki & web       | 78,785    | 8,837   | 11,313 |
| QA                        | PopQA           | wiki             | /         | /       | 14,267 |
| QA                        | SQuAD           | wiki             | 87,599    | 10,570  | /      |
| QA                        | MSMARCO-QA      | web              | 808,731   | 101,093 | /      |
| QA                        | NarrativeQA     | books and story  | 32,747    | 3,461   | 10,557 |
| QA                        | WikiQA          | wiki             | 20,360    | 2,733   | 6,165  |
| QA                        | WebQuestions    | Google Freebase  | 3,778     | /       | 2,032  |
| QA                        | AmbigQA         | wiki             | 10,036    | 2,002   | /      |
| QA                        | SIQA            | -                | 33,410    | 1,954   | /      |
| QA                        | CommonSenseQA   | -                | 9,741     | 1,221   | /      |
| QA                        | BoolQ           | wiki             | 9,427     | 3,270   | /      |
| QA                        | PIQA            | -                | 16,113    | 1,838   | /      |
| QA                        | Fermi           | wiki             | 8,000     | 1,000   | 1,000  |
| multi-hop QA              | HotpotQA        | wiki             | 90,447    | 7,405   | /      |
| multi-hop QA              | 2WikiMultiHopQA | wiki             | 15,000    | 12,576  | /      |
| multi-hop QA              | Musique         | wiki             | 19,938    | 2,417   | /      |
| multi-hop QA              | Bamboogle       | wiki             | /         | /       | 125    |
| multi-hop QA              | StrategyQA      | wiki             | 2290      | /       | /      |
| Long-form QA              | ASQA            | wiki             | 4,353     | 948     | /      |
| Long-form QA              | ELI5            | Reddit           | 272,634   | 1,507   | /      |
| Long-form QA              | WikiPassageQA   | wiki             | 3,332     | 417     | 416    |
| Open-Domain Summarization | WikiASP         | wiki             | 300,636   | 37,046  | 37,368 |
| multiple-choice           | MMLU            | -                | 99,842    | 1,531   | 14,042 |
| multiple-choice           | TruthfulQA      | wiki             | /         | 817     | /      |
| multiple-choice           | HellaSWAG       | ActivityNet      | 39,905    | 10,042  | /      |
| multiple-choice           | ARC             | -                | 3,370     | 869     | 3,548  |
| multiple-choice           | OpenBookQA      | -                | 4,957     | 500     | 500    |
| multiple-choice           | QuaRTz          | -                | 2696      | 384     | 784    |
| Fact Verification         | FEVER           | wiki             | 104,966   | 10,444  | /      |
| Dialog Generation         | WOW             | wiki             | 63,734    | 3,054   | /      |
| Entity Linking            | AIDA CoNll-yago | Freebase & wiki  | 18,395    | 4,784   | /      |
| Entity Linking            | WNED            | Wiki             | /         | 8,995   | /      |
| Slot Filling              | T-REx           | DBPedia          | 2,284,168 | 5,000   | /      |
| Slot Filling              | Zero-shot RE    | wiki             | 147,909   | 3,724   | /      |
| In-domain QA              | DomainRAG       | Web pages of RUC | /         | /       | 485    |

## 数据集保存格式

对于每个数据集，我们将数据集的每个split(train/dev/test)存储为一个`jsonl`文件，其中的每一行均为一个字典，格式如下:

```python
{
  'id': str,
  'question': str,
  'golden_answers': List[str],
  'metadata': dict
}
```

对于某些`test`集没有golden label标签的数据集，我们没有进行收集。

## 添加新数据集

如果需要添加新数据集，可以添加一个新数据集名称对应的文件夹(比如new_dataset)，并在文件夹内添加train.jsonl等文件，文件的格式需要和上面的数据集保存格式一致。


## 如何使用

我们推荐将需要使用的数据集直接下载到本地服务器，并按照下面的结构进行存储:

```md
FlashRAG_Data/
├── nq/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── triviaqa/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── hotpotqa/
    ├── train.jsonl
    ├── dev.jsonl
```

在使用FlashRAG时，只需要在`data_dir`中填入父文件夹对应的路径(例如: FlashRAG_Data/)，并填写对应的`dataset_name`和需要加载的`split`即可。