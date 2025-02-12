# 文档库准备

在RAG流程中，我们会根据query从文档库中检索相应的文档，其中涉及了两部分数据的准备:
1. **文档库**: 包含了所有我们希望检索的数据。
2. **索引**: 为了提升检索过程的效率，往往需要基于文档库构建索引，不同检索方法和模型分别对应不同的索引。

本节主要介绍文档库的准备流程。

## 支持的文档库格式

### 纯文本文档库

FlashRAG支持加载如下格式的`jsonl`文件作为文档库:
```python
{"id":"0", "contents": "..."}
{"id":"1", "contents": "..."}
```
其中，`id`为每条数据唯一对应的序列标识，`contents`包含了该条文档对应的内容。对于同时包含`title`和`text`的文档，我们推荐将`contents`设置为`{title}\n{text}`的形式。`id`和`contents`为必须设置的键，文档库文件同样可以包含其他额外的数据。

> [!TIP]
> 由于FlashRAG中默认会根据`\n`解析出title和text来构建prompt，我们推荐在没有title的文档库使用`\n{text}`来构建`contents`。

### 图文混合文档库

在多模态检索中，文档库的每一条数据可能同时包含图片和文本(比如图片和其对应的描述)，我们采用`parquet`格式来存储这类数据，其所包含的必要键值如下，可以根据需要添加额外的键值。

```python
{
    'id': str,
    'text': str,
    'image': str # 图片对应的字节流数据
}
```

> [!NOTE]
> 如果有多张图，可以先拼接成单张图再转换为字节流存放到`image`键下。

可以参考下面的代码将图片转换为字节流，并保存为`parquet`文件。

```python
import io
from PIL import Image
from datasets import Dataset

def image_to_bytes(image):
    """将Image列的PIL图片转换为字节"""
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")  # 可根据需要选择格式
        return buffer.getvalue()

# 加载图片
image = Image.open(image_path).convert('RGB')
# 图片转换为字节流保存
data = [{'id': '0', 'text': 'test sample', 'image': image_to_bytes(image)}]
# 存入parquet文件
data = Dataset.from_list(data)
data.to_parquet('sample_corpus.parquet')
```


加载这类文件可以使用`datasets`库自带的`load_dataset`函数，读取时会自动将图片的字节流转换为PIL格式。可以参考下面的代码:

```python
from datasets import load_dataset
data = load_dataset('parquet', data_files='sample_corpus.parquet', split='train')
data.cast_column('image', datasets.Image())
print(Data)
```

## 如何使用

![TIP]
> 以下内容仅针对纯文本模态。对于多模态文档库，直接填入对应路径即可。

### 使用现成文档库

FlashRAG提供了一个基于维基百科构建的文档库，托管在了数据集页面的`retrieval-corpus`路径下，只需下载并解压即可直接使用。该文档库基于Wikipedia 2018版本构建，对每个完整文档按照100个单词的粒度进行了切分。

### 基于维基百科构建新文档库

提供的维基百科2018版本可能相对较老，如果希望使用更新版本的维基百科作为检索源，可以参考下面的步骤进行构建。

步骤1：下载维基百科数据

使用如下命令下载所需的维基百科原始数据（XML格式），里面包含了每个词条的网页内容。

```bash
wget https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles.xml.bz2
```

你可以在该网站中找到其他版本的维基百科原始数据: [<u>网站链接</u>](https://archive.org/search?query=Wikimedia+database+dump&sort=-downloads).

步骤2: 运行处理脚本

FlashRAG提供了便捷脚本来处理原始的维基百科数据，只需要调整下面的参数即可将原始数据转换为支持的`jsonl`格式数据。脚本路径为`scripts/preprocess_wiki.py`, 其运行命令如下:

```bash
cd scripts
python preprocess_wiki.py --dump_path ../enwikinews-20240420-pages-articles.xml.bz2  \
                        --save_path ../output_corpus.jsonl \
                        --chunk_by sentence \
                        --chunk_size 512 \
                        --num_workers 1
```

其中，`chunk_by`指定了文本的分块方式，支持按照token、句子等各种切分方式。

### 使用自定义文档库

如需使用自定义的文档库，只需将文档按照上述格式处理好即可。


#### 自定义分块

在某些场景下可能需要对给定文档库进行分块后再使用，FlashRAG提供了便捷运行的分块脚本对文档库进行自定义分块。

给定一个包含以下格式的文档语料库JSONL文件，其中`contents`字段包含`"{title}\n{text}"`格式：

```jsonl
{ "id": 0, "contents": "..." }
{ "id": 1, "contents": "..." }
{ "id": 2, "contents": "..." }
```

运行如下命令可以直接得到分块后的文件:
```bash
cd scripts
python chunk_doc_corpus.py --input_path input.jsonl \
                          --output_path output.jsonl \
                          --chunk_by sentence \
                          --chunk_size 512
```

脚本涉及的参数说明如下:
* **input_path**: 输入的JSONL文件路径。
* **output_path**: 输出的JSONL文件路径。
* **chunk_by**: 分块方法。可以是token、word、sentence或recursive。
* **chunk_size**: 分块大小。
* **tokenizer_name_or_path**: 用于分块的分词器的名称或路径。


得到的新文件的格式为:
```json 
{ "id": 0, "doc_id": 0, "title": ..., "contents": ... }
{ "id": 1, "doc_id": 0, "title": ..., "contents": ... }
{ "id": 2, "doc_id": 0, "title": ..., "contents": ... }
```

其中, `doc_id`与原始文档`id`相同，`contents`是新生成的`jsonl`输出中的分块文档内容。



