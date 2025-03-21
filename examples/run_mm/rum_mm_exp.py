from flashrag.config import Config
from flashrag.utils import get_dataset
import argparse


def mathvista(args):
    clip_num = args.clip_num
    bm25_num = args.bm25_num
    mode = args.mode
    model_name = args.model_name
    config_dict = {
        "gpu_id": args.gpu_id,
        'dataset_name': 'mathvista',
        'test_sample_num': 10,
        'save_dir': 'result',
        'save_note': f'{model_name}-{mode}',
        'generator_model': model_name,
        'generation_params':{'max_new_tokens':256},
        'generator_max_input_len': 8192,
        'generator_batch_size': 1,
        'data_dir': "datasets",
        "use_multi_retriever": True,
        "multi_retriever_setting": {
            "merge_method": "concat",
            "retriever_list": [
                {
                    "retrieval_method": "bm25",
                    "corpus_path": "datasets/mathvista/train.parquet",
                    "index_path": "indexes/mathvista/bm25",
                    "retrieval_topk": bm25_num,
                    "bm25_backend": "bm25s",
                },
                {
                    "retrieval_method": "openai-clip-336",
                    "corpus_path": "datasets/mathvista/train.parquet",
                    "multimodal_index_path_dict": {
                        "image": "indexes/mathvista/openai-clip-vit-large-patch14-336_Flat_image.index",
                        "text": "indexes/mathvista/openai-clip-vit-large-patch14-336_Flat_text.index",
                    },
                    "retrieval_topk": clip_num,
                },
            ],
        },
        'metrics': ['acc', 'f1', 'em']
    }
    config = Config("./my_config.yaml", config_dict=config_dict)
    dataset = get_dataset(config)['test']

    from flashrag.pipeline import MMSequentialPipeline
    from flashrag.prompt import  MathVistaPromptTemplate

    if mode == 'no-ret':
        base_prompt_template = MathVistaPromptTemplate(config)
        pipeline = MMSequentialPipeline(config, prompt_template=base_prompt_template)
        dataset = pipeline.naive_run(dataset)
    else:
        base_prompt_template = MathVistaPromptTemplate(config)
        pipeline = MMSequentialPipeline(config, prompt_template=base_prompt_template)
        # dataset = pipeline.run(dataset, perform_modality_dict={'text': ['text']})
        dataset = pipeline.run(dataset)


def gaokao_mm(args):
    clip_num = args.clip_num
    bm25_num = args.bm25_num
    mode = args.mode
    model_name = args.model_name
    config_dict = {
        "gpu_id": args.gpu_id,
        'dataset_name': 'gaokao_mm',
        #'test_sample_num': 5,
        'save_dir': 'result',
        'save_note': f'{model_name}-{mode}',
        'generator_model': model_name,
        'generation_params':{'max_new_tokens':128},
        'generator_max_input_len': 4096,
        'generator_batch_size': 1,
        'data_dir': "datasets",
        "use_multi_retriever": True,
        "multi_retriever_setting": {
            "merge_method": "concat",
            "retriever_list": [
                {
                    "retrieval_method": "bm25",
                    "corpus_path": "datasets/gaokao_mm/train.parquet",
                    "index_path": "indexes/gaokao_mm/bm25/bm25",
                    "retrieval_topk": bm25_num,
                    "bm25_backend": "pyserini",
                },
                {
                    "retrieval_method": "chinese-clip",
                    "corpus_path": "datasets/gaokao_mm/train.parquet",
                    "multimodal_index_path_dict": {
                        "image": "indexes/gaokao_mm/chinese-clip-vit-large-patch14_Flat_image.index",
                        "text": "indexes/gaokao_mm/chinese-clip-vit-large-patch14_Flat_text.index",
                    },
                    "retrieval_topk": clip_num,
                },
            ],
        },
        'metrics': ['gaokao_acc']
    }
    config = Config("my_config.yaml", config_dict=config_dict)
    dataset = get_dataset(config)['test']

    from flashrag.pipeline import MMSequentialPipeline
    from flashrag.prompt import GAOKAOMMPromptTemplate
    from flashrag.utils import gaokaomm_pred_parse
    if mode == 'no-ret':
        zero_shot_prompt_template =  GAOKAOMMPromptTemplate(config, user_prompt="请你做一道{subject}选择题\n请你结合文字和图片一步一步思考,并将思考过程写在【解析】和<eoe>之间。{instruction}\n例如：{example}\n请你严格按照上述格式作答。\n题目如下：{question}")
        pipeline = MMSequentialPipeline(config, prompt_template=zero_shot_prompt_template)
        dataset = pipeline.naive_run(dataset, pred_process_func=gaokaomm_pred_parse)
    else:
        # base_prompt_template = NewPromptTemplate(config)
        base_prompt_template = GAOKAOMMPromptTemplate(config)
        pipeline = MMSequentialPipeline(config, prompt_template=base_prompt_template)
        # dataset = pipeline.run(dataset, pred_process_func=gaokaomm_pred_parse)
        dataset = pipeline.run(dataset, pred_process_func=gaokaomm_pred_parse,perform_modality_dict={'text': ['text']})
        
    
def mmqa(args):
    clip_num = args.clip_num
    bm25_num = args.bm25_num
    mode = args.mode
    model_name = args.model_name
    config_dict = {
        "gpu_id": args.gpu_id,
        'dataset_name': 'mmqa',
        #'test_sample_num': 5,
        'save_dir': 'result',
        'save_note': f'{model_name}-{mode}',
        'generator_model': model_name,
        'generation_params':{'max_new_tokens':128},
        'generator_max_input_len': 8192,
        'generator_batch_size': 1,
        'data_dir': "datasets",
        "use_multi_retriever": True,
        "multi_retriever_setting": {
            "merge_method": "concat",
            "retriever_list": [
                {
                    "retrieval_method": "bm25",
                    "corpus_path": "datasets/mmqa/train.parquet",
                    "index_path": "indexes/mmqa/bm25",
                    "retrieval_topk": bm25_num,
                    "bm25_backend": "pyserini",
                },
                {
                    "retrieval_method": "openai-clip",
                    "corpus_path": "datasets/mmqa/train.parquet",
                    "multimodal_index_path_dict": {
                        "image": "indexes/mmqa/openai-clip-vit-large-patch14_Flat_image.index",
                        "text": "indexes/mmqa/openai-clip-vit-large-patch14_Flat_text.index",
                    },
                    "retrieval_topk": clip_num,
                },
            ],
        },
        'metrics': ['acc', 'f1', 'em']
    }
    config = Config("my_config.yaml", config_dict=config_dict)
    dataset = get_dataset(config)['dev']

    from flashrag.pipeline import MMSequentialPipeline
    from flashrag.prompt import  MMPromptTemplate
    from flashrag.utils import gaokaomm_pred_parse

    if mode == 'no-ret':
        base_prompt_template = MMPromptTemplate(config)
        #zero_shot_prompt_template =  MMPromptTemplate(config, user_prompt="Answer the following question. Only give me the final answer.\nQuestion: {question}\nAnswer: ")
        pipeline = MMSequentialPipeline(config, prompt_template=base_prompt_template)
        dataset = pipeline.naive_run(dataset)
    else:
        base_prompt_template = MMPromptTemplate(config)
        pipeline = MMSequentialPipeline(config, prompt_template=base_prompt_template)
        # dataset = pipeline.run(dataset, perform_modality_dict={'text': ['text']})
        dataset = pipeline.run(dataset)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running mm exp")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_name", type=str,default='mmqa')
    parser.add_argument("--gpu_id", type=str,default='0,1')
    parser.add_argument("--mode", type=str,default="no-ret")
    parser.add_argument("--clip_num", type=int,default=1)
    parser.add_argument("--bm25_num", type=int,default=1)

    func_dict = {
        "mathvista":mathvista,
        "gaokao_mm":gaokao_mm,
        "mmqa":mmqa
    }
    args = parser.parse_args()

    func = func_dict[args.dataset_name]
    func(args)
