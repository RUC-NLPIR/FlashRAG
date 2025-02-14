# 精炼器

精炼器用于对检索结果进行后处理，包括压缩、精炼等。目前FlashRAG支持四种精炼器:

- 抽取式精炼: 对检索文档进行切分后，基于embedding模型保留与query相似度高得文本片段。支持各类embedding模型。
- 摘要式精炼: 基于生成模型对检索文档进行摘要和总结，支持BARTH和T5等`Seq2Seq`模型。
- 基于困惑度精炼: 利用LLM生成的困惑度对检索文档中的每个token判别重要性。支持`Selective-Context`和`LongLLMLingua`系列的方法。
- 基于知识图谱精炼: 基于**Trace**方法([论文链接](https://aclanthology.org/2024.findings-emnlp.496/))对检索文档进行精炼。主要思想是从检索文档中提取出事实三元组，并根据query选择合适的reasoning chain。


## 精炼器配置与加载


### 配置设置

不同类型的精炼器需要在配置文件中填写不同的参数，具体参数如下。

### 抽取式精炼器 (```ExtractiveRefiner```)

- `refiner_name`: `str`， 精炼器的名称，通常设置为`extractive`即可。
- `refiner_model_path`: `str`, 模型路径，填写需要使用的embedding模型的路径。
- `refiner_encode_max_length`: `int`, 最大输入长度。
- `refiner_pooling_method`: `str`，使用的池化方法，与retriever处的设置相同。默认为`mean`。
- `refiner_mini_batch_size`: `int`, 批量大小。

> [!TIP]
> `ExtractiveRefiner`的底层实现与检索器中的DenseRetriever一致，因此参数含义上可以参考检索器。

### 摘要式精炼器 (```AbstractiveRefiner```)

- `refiner_name`: `str`， 精炼器的名称，通常设置为`abstractive`即可。
- `refiner_model_path`: `str`, 模型路径，填写需要使用的摘要模型的路径。
- `refiner_max_input_length`: `int`, 最大输入长度。
- `refiner_max_output_length`: `int`, 最大输出长度。

### Selective-Context 精炼器 (```SelectiveContextRefiner```)

- `refiner_name`: `str`， 精炼器的名称，通常设置为`selective-context`即可。
- `refiner_model_path`: `str`, 模型路径，填写需要使用的GPT2模型的路径。
- `sc_config`: `dict`, 进行精炼的超参数。包含:
    - `reduce_ratio`: 希望压缩的信息的比例。
    - `reduce_level`: 压缩的信息的级别。默认为`phrase`， 可选`sent`, `token`。

### LLMLingua 精炼器 (```LLMLinguaRefiner```)

- `refiner_name`: `str`， 精炼器的名称，通常设置为`llmlingua`即可。
- `refiner_model_path`: `str`, 模型路径，填写需要用于计算困惑度的模型的路径，默认为`Llama2-7B`。
- `refiner_input_prompt_flag`: `bool`, 是否直接对prompt进行压缩。默认为`False`。
- `llmlingua_config`: `dict`, LLMLingua方法设计的超参数，包含:
    - `rate`: `float`, 压缩的比例，默认为`0.55`。
    - `rank_method`: 使用的压缩方法，默认为`longllmlingua`。

更多参数以及参数的具体解释可以参考`LLMLingua`官方的[参数说明](https://github.com/microsoft/LLMLingua/blob/main/DOCUMENT.md#function-call)。

### 基于知识图谱精炼器 (```TraceRefiner```)

- `refiner_name`: `str`，通常设置为`kg-trace`。
- `trace_config`: `dict`，Trace方法涉及的超参数，包括但不仅限于:
    - `num_examplars`: 使用的样例数量，默认为`3`。
    - `max_chain_length`: 构建的最大链长，默认为`4`。
    - `topk_triple_select`: 使用的候选triple的数量，默认为`5`。
    - `n_context`: 最终使用的reasoning chain的数量。

更多参数可以参考原始论文以及源码。

> [!NOTE]
> TraceRefiner并不类似于一个仅对检索文档做精炼/压缩的方法，中间过程涉及到使用检索器和生成器。建议参考其原始论文来理解。


### 加载

精炼器统一使用下面的代码进行加载，会自动根据填写的配置文件加载相应的精炼器:

```python
from flashrag.config import Config
from flahrag.utils import get_refiner
config = Config('myconfig.yaml')
refiner = get_refiner(config)
```

## 使用

精炼器的调用接口统一为`refiner.batch_run(dataset)`,需使用`flashrag.dataset.Dataset`作为输入，精炼器会使用其中的`question`和`retrieval_result`，并返回每个question对应的精炼后的文档。

参考代码如下:

```python
from flashrag.utils import get_refiner, get_dataset
config = Config('myconfig.yaml')
dataset = get_dataset(config)['test']
refiner = get_refiner(config)
# 返回值格式为: List[str]
refined_result = refiner.batch_run(dataset)
```

