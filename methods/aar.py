"""
Method name: AAR

Reference:
    Zichun Yu et al. "Augmentation-Adapted Retriever Improves Generalization of Language Models as Generic Plug-In"
    in ACL 2023.
    Official repo: https://github.com/OpenMatch/Augmentation-Adapted-Retriever
"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline


###### Specified parameters ######

# two types of checkpoint: ance / contriever
retrieval_method = "AAR-contriever"  
# download model checkpoint in 'OpenMatch/AAR-Contriever-KILT'
retriever_model2path = {"AAR-contriever": "OpenMatch/AAR-Contriever-KILT"}
# index path of this retriever
index_path = "..."  


config_dict = {
            'retrieval_method': retrieval_method,
            'retriever_model2path': retriever_model2path,
            'index_path': index_path
            }

# preparation
config = Config('my_config.yaml',config_dict)
all_split = get_dataset(config)
test_data = all_split['test']

pipeline = SequentialPipeline(config)
result = pipeline.run(test_data)
print(result)


