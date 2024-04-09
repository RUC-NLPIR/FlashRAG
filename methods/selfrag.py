"""
Reference:
    Akari Asai et al. " SELF-RAG: Learning to Retrieve, Generate and Critique through self-reflection"
    in ICLR 2024.
    Official repo: https://github.com/AkariAsai/self-rag
"""

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SelfRAGPipeline


# preparation
config = Config('my_config.yaml')
config_dict = {'generator_model_path': "selfrag/selfrag_llama2_7b",
               'use_vllm': True}

all_split = get_dataset(config, config_dict)
test_data = all_split['test']

pipeline = SelfRAGPipeline(config, threhsold=0.2, max_depth=2, beam_width=2, 
                           w_rel=1.0, w_sup=1.0, w_use=1.0,
                           use_grounding=True, use_utility=True, use_seqscore=True, ignore_cont=True,
                            mode='adaptive_retrieval')
result = pipeline.run(test_data)
# for long-form qa, can use long_form_run
# result = pipeline.long_form_run(test_data)
print(result)