## 1. 개요

이 문서는 표준 RAG 프로세스를 예로 들어 이 프로젝트의 다양한 구성과 기능을 소개하는 것을 목표로 합니다. 기존 방법의 재현과 개별 구성 요소의 자세한 사용법 등 복잡한 사용법에 대해서는 추후 추가 문서가 제공될 예정입니다.

표준 RAG 프로세스는 다음 세 가지 단계로 구성됩니다:
1. 사용자의 쿼리를 기반으로 지식 베이스에서 관련 문서를 검색합니다.
2. 검색된 문서와 원래 쿼리를 프롬프트에 포함시킵니다.
3. 프롬프트를 생성기(generator)에 입력합니다.

이 문서는 `E5`를 검색기(retriever)로, `Llama2-7B-Chat`을 생성기(generator)로 사용하여 RAG 프로세스를 시연할 것입니다.


## 2. 준비 사항

전체 RAG 프로세스를 원활하게 실행하려면 다음 다섯 가지 준비를 완료해야 합니다:

1. 프로젝트 및 종속 항목(dependencies) 설치
2. 필요한 모델 다운로드
3. 필요한 데이터셋 다운로드 ([토이 데이터셋](../examples/quick_start/dataset/nq) 제공)
4. 검색을 위한 문서 컬렉션 다운로드 ([토이 코퍼스](../examples/quick_start/indexes/general_knowledge.jsonl) 제공)
5. 검색을 위한 인덱스 빌드 ([토이 인덱스](../examples/quick_start/indexes/e5_Flat.index) 제공)

시작 시간을 절약하기 위해 토이 데이터셋, 문서 컬렉션 및 해당 인덱스를 제공합니다. 따라서 첫 두 단계를 완료하기만 하면 전체 프로세스를 성공적으로 실행할 수 있습니다.

### 2.1  프로젝트 및 종속 항목 설치

다음 명령을 사용하여 프로젝트 및 종속 항목을 설치하세요.

`vllm`, `fschat`, 또는 `pyserini` 패키지 설치 중 문제가 발생하면 `requirement.txt` 파일에서 해당 패키지를 주석 처리할 수 있습니다. 이러한 패키지는 특정 기능에 필요하지만, 일시적으로 생략해도 이 문서에서 설명하는 워크플로우에는 영향을 미치지 않습니다.

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e . 
```

### 2.2 모델 다운로드

다음 두 모델을 다운로드해야 합니다:

- E5-base-v2
- Llama2-7B-Chat

모델은 [Huggingface](https://huggingface.co/intfloat/e5-base-v2)에서 다운로드할 수 있습니다. 중국에 계신 경우 [hf-mirror](https://hf-mirror.com/) 플랫폼을 사용하여 다운로드하는 것이 좋습니다.

### 2.3 데이터셋 다운로드

데이터셋에는 쿼리 및 해당 표준 답변이 포함되어 있어 RAG 시스템의 효과를 평가할 수 있습니다.

간단히 하기 위해 NQ에서 17개의 데이터를 샘플링한 토이 데이터셋을 제공하며, 이는 [examples/quick_start/dataset/nq](../examples/quick_start/dataset/nq/)에 있습니다. 이후의 RAG 프로세스는 이 질문들에 대해 수행될 것입니다.

저희 리포지토리에는 많은 양의 처리된 벤치마크 데이터셋도 제공합니다. [huggingface datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)에서 다운로드하여 사용할 수 있습니다.

### 2.4 문서 컬렉션 다운로드

문서 컬렉션은 RAG 시스템의 외부 지식 원천 역할을 하는 많은 분할된(segmented) 단락을 포함합니다. 일반적으로 사용되는 문서 컬렉션은 매우 크기 때문에 (~5G 이상), [general knowledge dataset](https://huggingface.co/datasets/MuskumPillerum/General-Knowledge)을 토이 컬렉션으로 사용하며, 이는 [examples/quick_start/indexes/general_knowledge.jsonl](../examples/quick_start/indexes/general_knowledge.jsonl)에 위치해 있습니다.

> 문서 수가 적기 때문에 많은 쿼리가 관련 텍스트를 찾지 못할 수 있으며, 이는 최종 검색 결과에 영향을 미칠 수 있습니다.

전체 문서 컬렉션이 필요하면 [huggingface dataset](https://huggingface.co/datasets/ignore/FlashRAG_datasets)에서 다운로드하여 사용할 수 있습니다.

### 2.5 검색 인덱스 빌드

검색 효율성을 높이기 위해 검색 인덱스를 사전에 빌드해야 합니다. BM25 방법의 경우, 인덱스는 보통 역색인(우리 프로젝트에서는 디렉토리)입니다. 다양한 임베딩 방법의 경우 인덱스는 문서 컬렉션의 모든 텍스트 임베딩을 포함하는 Faiss 데이터베이스(.index 파일)입니다. **각 인덱스는 하나의 코퍼스와 하나의 검색 방법에 대응됩니다**. 즉 새로운 임베딩 모델을 사용할 때마다 인덱스를 다시 빌드해야 합니다.

여기서는 E5-base-v2와 앞서 언급한 토이 코퍼스를 사용하여 빌드한 [토이 인덱스](../examples/quick_start/indexes/e5_Flat.index)를 제공합니다.

자신의 검색 모델과 문서를 사용하려면 [index building document](./building-index.md)를 참조하여 인덱스를 빌드할 수 있습니다.


## 3. Running the RAG Process

다음 단계에서는 각 단계를 세분화하고 해당 코드를 시연할 것입니다. 전체 코드는 마지막에 제공되거나 [simple_pipeline.py](../examples/quick_start/simple_pipeline.py) 파일을 참조할 수 있습니다.

### 3.1 Loading the Config

먼저 `Config`를 로드하고 앞서 다운로드한 항목들의 경로를 채워야 합니다.

`Config`는 실험의 모든 경로와 하이퍼파라미터를 관리합니다. FlashRAG에서는 다양한 매개변수를 yaml 파일 또는 Python 딕셔너리를 통해 Config에 전달할 수 있습니다. 전달된 매개변수는 기본 내부 매개변수를 대체합니다. 자세한 매개변수 정보와 기본값은 [`basic_config.yaml`](../flashrag/config/basic_config.yaml)을 참조할 수 있습니다.

여기서는 딕셔너리를 통해 경로를 직접 전달합니다.

```python
from flashrag.config import Config

config_dict = { 
    'data_dir': 'dataset/',
    'index_path': 'indexes/e5_Flat.index',
    'corpus_path': 'indexes/general_knowledge.jsonl',
    'model2path': {'e5': <retriever_path>, 'llama2-7B-chat': <generator_path>},
    'generator_model': 'llama2-7B-chat',
    'retrieval_method': 'e5',
    'metrics': ['em', 'f1', 'acc'],
    'retrieval_topk': 1,
    'save_intermediate_data': True
}

config = Config(config_dict=config_dict)
```

### 3.2 데이터셋 및 파이프라인 로드

다음으로 데이터셋과 파이프라인을 불러와야 합니다.

데이터셋은 앞서 설정된 config를 통해 자동으로 불러올 수 있으며, 우리는 해당 테스트셋을 선택하기만 하면 됩니다.

파이프라인 로딩은 원하는 RAG 프로세스에 따라 적절한 파이프라인을 선택해야 합니다. 여기서는 앞서 언급한 표준 RAG 프로세스를 위해 `SequentialPipeline`을 선택합니다.
파이프라인은 자동으로 해당 구성 요소(검색기 및 생성기)를 불러오고 다양한 초기화를 완료합니다.

```python
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline

all_split = get_dataset(config)
test_data = all_split['test']
pipeline = SequentialPipeline(config)
```

### 3.3 Running the RAG Process

위의 단계를 완료한 후, 파이프라인의 `.run` 메서드를 호출하여 데이터셋에서 RAG 프로세스를 실행하고 평가 결과를 생성하면 됩니다. 이 메서드는 중간 결과와 최종 결과를 포함하는 데이터셋을 반환하며, pred 속성에 모델의 예측이 담깁니다.

토이 문서 컬렉션과 인덱스를 제공했기 때문에 결과가 상대적으로 좋지 않을 수 있습니다. 더 나은 결과를 위해 자신만의 문서 컬렉션과 인덱스를 사용하는 것을 고려해보세요.

프로세스가 완료되면 모든 결과는 현재 실험에 해당하는 폴더에 저장되며, 여기에는 각 쿼리에 대한 검색 및 생성 결과, 전체 평가 점수 등이 포함됩니다.

전체 코드는 다음과 같습니다:

```python
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline

config_dict = { 
                'data_dir': 'dataset/',
                'index_path': 'indexes/e5_Flat.index',
                'corpus_path': 'indexes/general_knowledge.jsonl',
                'model2path': {'e5': <retriever_path>, 'llama2-7B-chat': <generator_path>},
                'generator_model': 'llama2-7B-chat',
                'retrieval_method': 'e5',
                'metrics': ['em','f1','acc'],
                'retrieval_topk': 1,
                'save_intermediate_data': True
            }

config = Config(config_dict = config_dict)

all_split = get_dataset(config)
test_data = all_split['test']
pipeline = SequentialPipeline(config)

output_dataset = pipeline.run(test_data,do_eval=True)
print("---generation output---")
print(output_dataset.pred)
```

