# 配置设置

FlashRAG通过实现的`Config`类对实验所需的所有参数进行管理。`Config`类支持两种类型的参数配置: **YAML配置文件**和**参数字典**，实验中需要的参数可以通过任一方式进行传递 (**参数字典优先级更高**)。

一个具体的例子如下:
```python
from flashrag.config import Config
config_dict = {'generator_model': 'llama2-7B'}
config = Config(
    config_file_path='myconfig.yaml', 
    config_dict=config_dict
)
model_name = config['generator_model']
# 期待输出: 'llama2-7B'
print(model_name)
```

## 使用逻辑

### 配置文件

配置文件应为 YAML 格式，其内部使用键值对的方式存储需要的配置设置，用户应根据 YAML 语法填写相应的参数。在我们的库中，我们提供了一个模板文件供用户参考 (`flashrag\config\basic_config.yaml`)，并附有注释解释每个参数的具体含义。

使用配置文件的代码如下：
```python
from flashrag.config import Config
config = Config(config_file_path='myconfig.yaml')
```

> [!NOTE]
> 初始化`Config`的时候可以不指定`yaml`文件，这种情况下会加载位于`flashrag\config\basic_config.yaml`下的配置文件作为默认配置。

### 参数字典

另一种方式是通过 Python 字典设置配置，字典中的键为参数名称，值为参数值。与使用文件相比，这种方法更为灵活。

使用参数字典的代码如下：
```python
from flashrag.config import Config
config_dict = {'generator_model': 'llama2-7B'}
config = Config(config_dict=config_dict)
```

### 优先级

在 FlashRAG 中，我们支持组合两种方法。

配置方法的优先级为：参数字典 > 配置文件 > 默认设置

默认设置存放在 basic_config.yaml 中。



