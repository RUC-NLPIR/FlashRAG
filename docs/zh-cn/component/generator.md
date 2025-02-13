# 生成器（Generator）

生成器（Generator）模块是FlashRAG框架中的一个核心组件，主要用于根据用户的输入生成文本内容。在RAG（检索增强生成）任务中，生成器接收由检索器提供的相关文档或信息，并根据这些信息生成最终的答案或文本输出。

## 支持的生成器列表

FlashRAG通过集成`VLLM`,`FastChat`等先进推理框架来支持市面上各类生成模型。同时还通过兼容`Openai`的api接口来支持云端的模型调用。由于每个框架支持的模型类型不一样，下面按照推理框架的不同进行列举。


### 原生Transformers框架 / FastChat框架

原生Transformers框架是实现各类模型的基础框架，主要支持Encoder-Decoder结构的模型和CausalLM结构的模型推理。

- Encoder-Decoder结构: 支持BART, T5以及类似结构模型的推理，并支持FID (Fusion-in-Decoder)的推理方式 (需要base模型是T5)。
- CausalLM结构: 基于Transformers框架的`AutoModelForCausalLM`接口进行模型加载，支持Llama,Qwen,Mistral等主流LLM推理，支持Lora模型的加载。

> [!NOTE] 
> 如果出现模型生成内容异常的情况，可能是tokenizer需要进行特殊设置，可以在`generator/generator.py`下的`HFCausalLMGenerator._load_model`函数添加自定义的修改，或在github上创建issue，我们会尽快解决。

由于FastChat框架底层基于Transformers的接口进行模型加载，其支持范围与Transformers框架相同，仅在模型推理和加载到GPU过程中存在一些差异。

#### 多模态模型

目前，我们基于Transformers框架实现了三类多模态生成器:

- Qwen2VL
- Intern2VL
- Llava, Llava-next

### VLLM

VLLM框架基本支持市面上所有LLM的推理，具体模型列表可以参考: [VLLM文档](https://docs.vllm.ai/en/latest/models/supported_models.html)>

### Openai

Openai接口支持各类基于API进行调用的LLM。

## 生成器的配置与加载

生成器的配置可以参考[配置文件相关说明](../configuration/generator.md)。

要使用生成器，只需按照以下步骤进行加载：

```python
from flashrag.config import Config
from flashrag.utils import get_generator

# 加载自定义的配置文件，配置生成器参数
config = Config('my_config.yaml')
generator = get_generator(config)
```


## 使用生成器

### 纯文本生成器 (LLM)

对于基于各种框架实现的生成器，统一使用`generate`方法进行文本生成。为了适应某些算法中需要推理过程中的额外信息(比如hidden state, logits等)，不同框架的生成器接口有略微的不同。

#### 输入参数

通用参数:

- `input_list`: List[str], 输入的prompt列表，每个item为添加好特殊token后的字符串。

> [!NOTE]
> 由于部分chat模型的推理涉及到添加特殊token(用于区分chat/system message等功能)，直接输入原始prompt进行推理可能会导致模型输出的效果不理想。推荐使用`prompt_template`进行prompt的构造，并将构造后的prompt作为这里`generator`的输入。具体可以参考prompt构建的章节。


- `params`: dict, 推理参数，比如`temperature`, `top_k`等推理时涉及的超参数。需要注意的是不同框架的推理参数名称可能存在差异，需要根据具体框架进行设置。接口中的params优先级大于配置文件中设置的`generation_params`。

> [!TIP]
> FlashRAG中已经对大部分常见参数进行了兼容，可以不用根据框架进行特殊设置。


其他参数:

#### HF，FastChat框架

- batch_size: int， 确定生成器推理时的批处理大小。
- return_scores: bool, 是否返回模型推理的序列概率分数。打开后返回值为元组:`(responses, scores)`。默认为`False`，返回生成的文本。
- return_dict: bool, 是否返回包含推理过程中生成的其他信息的dict。打开后返回值为`dict`类型。默认为`False`，返回生成的文本。


#### VLLM框架

- return_raw_output: bool, 打开后返回值为VLLM框架的原始输出类。默认为`False`，返回生成的文本。
- return_scores: bool, 是否返回模型推理的序列概率分数。打开后返回值为元组:`(responses, scores)`。默认为`False`，返回生成的文本。

#### 返回结果

默认情况下，`generate`方法返回一个列表，每个元素为生成的文本结果。其余设置下的返回值可以参考对应的框架的文档。

### 多模态生成器

> [!TIP]
> 目前仅支持HF框架下的多模态生成器，后续会添加VLLM等其他框架的支持。

目前FlashRAG支持图文模态的混合输入。多模态检索器同样采用`generate`作为生成的接口，其输入参数如下。

#### 输入参数

多模态生成器的输入包括三个参数: `input_list`, `batch_size`,`params`。

输入函数: `generate(input_list, batch_size, **params)`

- `input_list`: List[List[dict]], 输入的prompt列表，每个元素对应该item的输入。
由于输入涉及到图片和文本的混合输入，我们采取下面的输入格式。`input_list`中的每个item具体格式如下:

```python
[
    {"role" : "system", "content": str or List[dict]},
    {"role" : "user", "content": str or List[dict]}
]
```
其中可以包含若干个`role`为`user`或`system`的message块 (**一般一个system message + 一个user message 或 一个user message**)。每个message块的`content`可以为字符串或者字典。

如果`content`需要输入图片，则需要用下面的字典格式来构建：
```python
# content的结构
[
    {"type": "image", "image": "sample_pic.png"},
    {"type": "text", "text": "请你描述一下这张图片，用一句话"}
]
```

每个`content`可以包含若干个`image`和一个`text`块。需要注意的是，需要确认使用的VLM是否支持多个`image`的输入。对于`image`输入，支持使用`pil.Image`对象或者本地图片路径或者url链接。对于`text`输入，支持直接传入文本。

> [!TIP]
> 与纯文本模态的Generator的不同，多模态生成器不需要使用`prompt_template`进行构造，可以直接构造成现在支持的格式。

一个输入完整的输入示例如下:

```python
# 输入两条prompt进行处理，每条包含一个图片
messages = [
        # query1
        [{
            "role": "user", 
            "content": [
                {"type": "image", "image": "sample1.png"},
                {"type": "text", "text": "请你描述一下这张图片，一句话"},
            ],
        }],
        # query2
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": "sample_2.png"},
                {"type": "image", "image": "sample_3.png"},
                {"type": "text", "text": "请你比较一下这两张图片"},
            ],
        }],
    ]
# 结果包含两个case的回复
output = generator.generate(messages, batch_size=1)
```

- `**params`: dict, 包含推理所需的参数，比如`temperature`, `top_k`等推理时涉及的超参数。接口中的params优先级大于配置文件中设置的`generation_params`。



