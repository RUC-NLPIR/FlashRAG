from flashrag.config import Config
from flashrag.utils import get_dataset
import argparse

def naive(args):
    save_note = 'naive'
    config_dict = {'save_note': save_note,
                'gpu_id':args.gpu_id,
                'dataset_name':args.dataset_name}

    from flashrag.pipeline import SequentialPipeline
    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    import torch
    print(torch.cuda.device_count())
    assert False
    pred_process_fun = lambda x: x.split("\n")[0]
    pipeline = SequentialPipeline(config)
    
    result = pipeline.run(test_data)

def zero_shot(args):
    save_note = 'zero-shot'
    config_dict = {'save_note': save_note,
                'gpu_id':args.gpu_id,
                'dataset_name':args.dataset_name}

    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    pred_process_fun = lambda x: x.split("\n")[0]
    pipeline = SequentialPipeline(config)
    result = pipeline.naive_run(test_data)

def aar(args):
    # two types of checkpoint: ance / contriever
    #retrieval_method = "AAR-contriever"  # AAR-ANCE
    # index path of this retriever
    retrieval_method = args.method_name
    if 'contriever' in retrieval_method:
        index_path = "aar-contriever_Flat.index" 
    else:
        index_path = "aar-ance_Flat.index" 

    retriever_model2path = {"AAR-contriever": "model/AAR-Contriever-KILT",
                            "AAR-ANCE": "model/AAR-ANCE"}
    model2pooling = {"AAR-contriever": "mean",
                    "AAR-ANCE": "cls"}
    save_note = retrieval_method
    config_dict = {
                'retrieval_method': retrieval_method,
                'retriever_model2path': retriever_model2path,
                'index_path': index_path,
                'model2pooling': model2pooling,
                'save_note': save_note,
                'gpu_id':args.gpu_id,
                'dataset_name':args.dataset_name
                }

    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    pred_process_fun = lambda x: x.split("\n")[0]
    pipeline = SequentialPipeline(config)
    #result = pipeline.run(test_data, pred_process_fun=pred_process_fun)
    result = pipeline.run(test_data)

def llmlingua(args):
    refiner_name = "longllmlingua" # 
    refiner_model_path = "model/llama-2-7b-hf"

    config_dict = {
        'refiner_name': refiner_name,
        'refiner_model_path': refiner_model_path,
        'llmlingua_config':{
            'rate': 0.55,
            'condition_in_question': 'after_condition',
            'reorder_context': 'sort',
            'dynamic_context_compression_ratio': 0.3,
            'condition_compare': True,
            'context_budget': "+100",
            'rank_method': 'longllmlingua'
        },
        'refiner_input_prompt_flag': False,
        'save_note':'longllmlingua',
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name
    }

    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data)

def recomp(args):
    # ###### Specified parameters ######
    refiner_name = "recomp-abstractive" # recomp-extractive
    model_dict = {'nq':"model/recomp_nq_abs",
                  'triviaqa':"model/recomp_tqa_abs",
                  'hotpotqa':'model/recomp_hotpotqa_abs'}

    refiner_model_path = model_dict.get(args.dataset_name, None)
    refiner_max_input_length = 1024
    refiner_max_output_length = 512
    # parameters for extractive compress
    refiner_topk = 5
    refiner_pooling_method = 'mean'
    refiner_encode_max_length = 256


    config_dict = {
        'refiner_name': refiner_name,
        'refiner_model_path': refiner_model_path,
        'refiner_max_input_length': refiner_max_input_length,
        'refiner_max_output_length': refiner_max_output_length,
        'refiner_topk': 5,
        'refiner_pooling_method': refiner_pooling_method,
        'refiner_encode_max_length': refiner_encode_max_length,
        'save_note': refiner_name,
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name
    }


    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data)

def sc(args):
    refiner_name = "selective-context"  
    refiner_model_path = "model/gpt2"

    config_dict = {
        'refiner_name': refiner_name,
        'refiner_model_path': refiner_model_path,
        'sc_config':{
            'reduce_ratio': 0.5
        },
        'save_note': 'selective-context',
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name
    }


    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data)

def retrobust(args):
    model_dict = {'nq':'model/llama-2-13b-peft-nq-retrobust',
                  '2wiki':'model/llama-2-13b-peft-2wikihop-retrobust'}
    if args.dataset_name == 'triviaqa':
        lora_path = model_dict['nq']
    elif args.dataset_name == 'hotpotqa':
        lora_path = model_dict['2wiki']
    else:
        lora_path = model_dict.get(args.dataset_name,None)
    config_dict = {'save_note':'Ret-Robust',
               'generator_model': 'llama2-13B',
               'generator_lora_path': lora_path,
               'generation_params':{"max_tokens":100},
               'gpu_id':args.gpu_id,
               'use_vllm':False,
               'generator_max_input_len': 4096,
               'dataset_name':args.dataset_name}
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SelfAskPipeline
    from flashrag.utils import selfask_pred_parse
    pipeline = SelfAskPipeline(config, max_iter=5, single_hop=False)
    # use specify prediction parse function 
    result = pipeline.run(test_data, pred_process_fun=selfask_pred_parse)

def sure(args):
    config_dict = {
            'save_note': 'SuRe',
            'gpu_id':args.gpu_id,
            'dataset_name':args.dataset_name
            }
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SuRePipeline
    pipeline = SuRePipeline(config)
    pred_process_fun = lambda x: x.split("\n")[0]
    result = pipeline.run(test_data)

def replug(args):
    save_note = 'replug'
    config_dict = {'save_note': save_note,
                'gpu_id':args.gpu_id,
                'dataset_name':args.dataset_name}

    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    pred_process_fun = lambda x: x.split("\n")[0]

    from flashrag.pipeline import REPLUGPipeline
    pipeline = REPLUGPipeline(config)
    result = pipeline.run(test_data)

def skr(args):
    judger_name = 'skr'
    model_path = 'model/sup-simcse-bert-base-uncased'
    training_data_path = './sample_data/skr_training.json'

    config_dict = {
        'judger_name': judger_name,
        'judger_model_path': model_path,
        'judger_training_data_path': training_data_path,
        'judger_topk': 5,
        'save_note': 'skr',
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name
    }


    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import ConditionalPipeline
    pipeline = ConditionalPipeline(config)
    result = pipeline.run(test_data)

def selfrag(args):
    config_dict = {'generator_model':'selfrag-llama2-7B',
                'generator_model_path': 'model/selfrag_llama2_7b',
               'use_vllm': True,
               'save_note':'self-rag',
               'gpu_id':args.gpu_id,
               'generation_params':{'max_new_tokens':100,'temperature':0.0,'top_p':1.0,'skip_special_tokens':False},
                'dataset_name':args.dataset_name}
    config = Config('my_config.yaml', config_dict)


    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SelfRAGPipeline
    pipeline = SelfRAGPipeline(config, threhsold=0.2, max_depth=2, beam_width=2, 
                            w_rel=1.0, w_sup=1.0, w_use=1.0,
                            use_grounding=True, use_utility=True, use_seqscore=True, ignore_cont=True,
                                mode='adaptive_retrieval')
    result = pipeline.run(test_data, batch_size=256)

def flare(args):
    config_dict={'save_note':'flare', 'gpu_id':args.gpu_id, 'use_vllm':True,
                'dataset_name':args.dataset_name}
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]


    from flashrag.pipeline import FLAREPipeline
    pipeline = FLAREPipeline(config)
    result = pipeline.run(test_data)

def iterretgen(args):
    iter_num = 3
    config_dict = {'save_note': 'iter-retgen',
                'gpu_id':args.gpu_id,
                'dataset_name':args.dataset_name}
    # preparation
    config = Config('my_config.yaml', config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import IterativePipeline
    pipeline = IterativePipeline(config, iter_num=iter_num)
    result = pipeline.run(test_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Running exp")
    parser.add_argument('--method_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--dataset_name',type=str)
    parser.add_argument('--gpu_id', type=str)

    func_dict = {
        'AAR-contriever': aar,
        'AAR-ANCE': aar,
        'naive': naive,
        'zero-shot': zero_shot,
        'llmlingua': llmlingua,
        'recomp': recomp,
        'selective-context': sc,
        'ret-robust': retrobust,
        'sure': sure,
        'replug': replug,
        'skr': skr,
        'selfrag': selfrag,
        'flare': flare,
        'iterretgen': iterretgen
    }

    args = parser.parse_args()

    func = func_dict[args.method_name]
    func(args)

