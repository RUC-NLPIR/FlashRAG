# 环境设置

环境设置主要包含与组件无关的一些全局设置，包括自动路径设置，全局设置和评测设置。

## 自动路径设置

为了方便管理各种模型路径和索引路径，我们引入了 `model2path`、`model2pooling` 和 `method2index`。用户只需在这里填写相应的名称和路径，即可自动加载所需路径。

例如，在 model2path 中设置了 llama2-7B 的路径后，可以直接在`generator_model`里面填写`llama2-7B`,无需再填写`generator_model_path`，可以简化各类路径的管理。
```yaml
model2path:
    llama2-7B: 'my_path'
generator_model: 'llama2-7B'
```

如果在`model2path`中没有找到`generator_model`对应的键，则需要填写`generator_model_path`作为模型路径。

其识别关系如下:

* **model2path**: 识别`generator_model`，`retrieval_method`和`rerank_model_name`，并自动填写对应模型路径。
* **model2pooling**: 识别`retrieval_method`，并填入`retrieval_pooling_method`。(当前版本pooling已经能够自动填写，不需要设置)
* **index2path**: 识别`retrieval_method`，并填入`index_path`。

## 全局设置

全局设置包括了实验环境以及数据集加载的各种配置。涉及全局设置的参数如下:

```yaml
# 数据集加载路径以及实验保存路径
data_dir: "dataset/"
save_dir: "output/"

gpu_id: "0,1,2,3"
dataset_name: "nq" # 数据集名称
split: ["test"] # 需要加载的数据集子集

test_sample_num: ~ # 要测试的样本数（仅在 dev/test 分割中有效），如果为 None，则测试所有样本
random_sample: False # 是否随机采样测试样本

# 随机种子
seed: 2024

# 是否保存中间数据
save_intermediate_data: True
save_note: "experiment"
```

其中重要参数的含义如下：
* `data_dir`:  存放所有数据集的路径，加载的数据集需存放在`data_dir/dataset_name路径下
* `split`：指定要加载的数据集子集(train/dev/test), 所有被指定的子集都会被加载。
* `save_note`, `save_dir`：每次初始化`Config`都将在 `save_dir` 中创建一个新文件夹，并在文件夹名称中添加 `save_note` 作为标记。文件夹会保存实验涉及的参数、中间数据和评测结果。
* `save_intermediate_data`：如果启用，中间结果将记录在实验文件夹中，包括检索内容、生成内容等。
* gpu_id：指定实验使用哪些 GPU，支持多 GPU。


## 评测设置

评估设置部分定义了实验中用来衡量模型表现的各种评估指标以及其他相关配置，帮助在实验过程中进行结果评估。以下是该部分配置文件的详细说明：

```yaml
# 用到的评测指标
metrics: ["em", "f1", "acc", "precision", "recall", "input_tokens"]
# 评测指标中涉及的特殊设置
metric_setting:
  retrieval_recall_topk: 5
  tokenizer_name: "gpt-4"
# 是否保存最终的评测分数
save_metric_score: True 
```

参数含义如下:

* metrics: 评估方法时使用的评测指标指标列表。目前支持的评测指标包括：
     - `em`: exact match，即模型输出结果与标准答案是否完全一致。
     - `acc`: accuracy，即模型输出结果是否包含正确答案。
     - `f1`: F1 分数，衡量输出答案和标准答案的token-level分数。
     - `precision`: Precision 分数，衡量输出答案和标准答案的token-level 分数。
     - `recall`: recall 分数，衡量输出答案和标准答案的token-level分数。
     - `input_tokens`: 输入模型的 tokens 的数量，用于记录每个样本的输入长度。
     - `retrieval_recall`: 检索的召回率，衡量检索文档是否包含正确答案。
     - `retrieval_precision`: 检索的精确性
     - `rouge-1`, `rouge-2`, `rouge-l`
     - `zh_rouge-1`, `zh_rouge-2`, `zh_rouge-l`: 中文版本
     - `bleu`
     - `gaokao_acc`: 评估GAOKAO-MM数据集的各科目Accuracy
     - `llm_judge`: 基于LLM对生成答案进行打分，需要额外配置`llm_judge_setting`。

* metric_setting: 该部分定义了与评估指标相关的额外设置：
     - `retrieval_recall_topk`: 在检索任务中，指定需要考虑的召回结果数量，这里设置为 5，意味着评估时考虑前 5 个候选结果。
     - `tokenizer_name`: 用于计算输入token数量，可指定openai系列的分词器以及HF支持的各类tokenizer。

* save_metric_score: 是否将评估结果保存到txt文件中。设置为 `True` 表示会将每个评估指标的得分保存在实验文件夹下的`metric_score.txt`文件中。

