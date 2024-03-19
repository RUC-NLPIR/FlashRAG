import re
import os
import yaml
import json
import random
import copy
import importlib
import sys

class Config:
    def __init__(
        self, config_file_path = None, config_dict = {}
    ):
        self.yaml_loader = self._build_yaml_loader()
        self.file_config = self._load_file_config(config_file_path)
        self.variable_config = config_dict

        self.external_config = self._merge_external_config()

        self.internal_config = self._get_internal_config()

        self.final_config = self._get_final_config()

        # TODO: 处理文件夹路径, 包括绝对路径和相对路径
        
        self._check_final_config()
        self._set_additional_key()

        self._init_device()
        self._set_seed()


    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader

    def _load_file_config(self, config_file_path:str):
        file_config = dict()
        if config_file_path:
            with open(config_file_path, "r", encoding="utf-8") as f:
                file_config.update(
                    yaml.load(f.read(), Loader=self.yaml_loader)
                )
        return file_config

    def _merge_external_config(self):
        external_config = dict()
        external_config.update(self.file_config)
        external_config.update(self.variable_config)
        
        return external_config

    def _get_internal_config(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        init_config_path = os.path.join(current_path, "basic_config.yaml")    
        internal_config = self._load_file_config(init_config_path)

        return internal_config
        
    def _get_final_config(self):
        final_config = dict()
        final_config.update(self.internal_config)
        final_config.update(self.external_config)

        return final_config

    def _check_final_config(self):
        # check split
        split = self.final_config['split']
        if split is None:
            split = ['train','dev','test']
        if isinstance(split, str):
            split = [split]
        self.final_config['split'] = split

    def _init_device(self):
        gpu_id = self.final_config['gpu_id']
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            import torch
            self.final_config['device'] = torch.device('cuda')
        else:
            import torch
            self.final_config['device'] = torch.device('cpu')

    def _set_additional_key(self):
        # set dataset
        dataset_name = self.final_config['dataset_name']
        data_dir = self.final_config['data_dir']
        self.final_config['dataset_path'] = os.path.join(data_dir, dataset_name)

        # set model path
        retrieval_method = self.final_config['retrieval_method']
        retriever_model2path = self.final_config['retriever_model2path']
        model2pooling = self.final_config['model2pooling']
        method2index = self.final_config['method2index']

        generator_model = self.final_config['generator_model']
        generator_model2path = self.final_config['generator_model2path']


        if self.final_config['index_path'] is None:
            self.final_config['index_path'] = method2index.get(retrieval_method, None)

        self.final_config['retrieval_model_path'] = retriever_model2path.get(retrieval_method, retrieval_method)
        # TODO: not support when `retrieval_model` is path
        self.final_config['retrieval_pooling_method'] = model2pooling.get(retrieval_method, "pooler")

        self.final_config['generator_model_path'] = generator_model2path.get(generator_model, generator_model)
        
    def _set_seed(self):
        import torch
        import numpy as np
        seed = self.final_config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    

    
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config[key] = value

    def __getattr__(self, item):
        if "final_config" not in self.__dict__:
            raise AttributeError(
                f"'Config' object has no attribute 'final_config'"
            )
        if item in self.final_config:
            return self.final_config[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        return self.final_config.get(item)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config
    
    def __repr__(self):
        return self.final_config.__str__()