# 生成器设置

- **framework**  
  设置LLM的推理框架。支持以下选项：
  - `hf`：使用HuggingFace原生的transformers框架，不支持多卡并行。
  - `vllm`：VLLM框架。
  - `fschat`：使用FastChat推理框架。
  - `openai`：OpenAI 的推理框架，适用于使用OpenAI格式的API进行推理。

- **generator_model**  
  指定生成模型的名称或路径。该参数用于指定要使用的生成模型。对于大多数模型，您需要提供模型的名称或完整路径。
  示例值：`"llama3-8B-instruct"`。

- **openai_setting**  
  仅在使用 OpenAI 推理框架时有效，配置与 OpenAI API 相关的参数，包括但不仅限于:
  - **api_key**：设置 OpenAI API 密钥，用于身份验证。
  - **base_url**：设置 OpenAI API 的基础 URL，通常为 OpenAI 的默认 API 地址，除非您使用自定义部署。

> [!TIP]
> 支持的参数可以参考Openai的Client加载的参数。

- **generator_model_path**  
  设置生成模型的路径。如果使用本地模型，提供模型文件对应的文件夹路径。

- **generator_max_input_len**  
  设置生成模型的最大输入长度。此值控制生成模型能够处理的最大输入文本的长度。如果输入文本超出此长度，将会被截断。

- **generator_batch_size**  
  设置生成时的批次大小。批次大小决定了一次处理多少个生成任务。对于 VLLM 框架，该参数无效。

- **use_fid**  
  是否使用 FID (fusion-in-decoder)，仅适用于T5模型。

- **gpu_memory_utilization**  
  设置 GPU 内存的使用比例，仅用于VLLM框架。该参数决定生成过程中 GPU 的内存使用量。设置为 `0.85` 表示将使用 GPU 总内存的 85% 来进行生成。合理设置此参数可确保在 GPU 上运行时不会出现内存溢出。

- **generation_params**  
  生成模型的其他参数。可以根据需要调整以下选项：
  - `max_tokens`：指定生成的最大 token 数量。默认值为 `32`，可以根据需要增加或减少生成文本的长度。

  其他常见生成参数包括：
  - `do_sample`：是否启用采样。如果设置为 `True`，模型将进行随机采样生成。默认情况下可以设置为 `False`，表示使用贪心策略。
  - `temperature`：控制生成文本的随机性。温度较低时生成的文本会更为确定，较高时会更具多样性。默认值为 `1.0`。
  - `top_p`：与 `temperature` 配合使用，控制生成文本的多样性。它表示采样时仅考虑概率累积大于某个阈值的候选token。默认值为 `1.0`。

> [!NOTE]
> 尽管FlashRAG中对各个推理框架的大部分参数设置进行了对齐，但有些参数可能在特定模型或推理框架中存在差异。如果遇到相关错误，可以查阅推理框架的相关文档或在issue中进行反馈。
