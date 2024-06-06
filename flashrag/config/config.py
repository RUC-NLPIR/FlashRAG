import re
import os
import yaml
import random
import datetime

class Config:
    def __init__(self, config_file_path = None, config_dict = {}):

        self.yaml_loader = self._build_yaml_loader()
        self.file_config = self._load_file_config(config_file_path)
        self.variable_config = config_dict

        self.external_config = self._merge_external_config()

        self.internal_config = self._get_internal_config()

        self.final_config = self._get_final_config()

        self._check_final_config()
        self._set_additional_key()

        self._init_device()
        self._set_seed()
        self._prepare_dir()


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

    @staticmethod
    def _update_dict(old_dict: dict, new_dict: dict):
        # Update the original update method of the dictionary:
        # If there is the same key in `old_dict` and `new_dict`, and value is of type dict, update the key in dict

        same_keys = []
        for key,value in new_dict.items():
            if key in old_dict and isinstance(value, dict):
                same_keys.append(key)
        for key in same_keys:
            old_item = old_dict[key]
            new_item = new_dict[key]
            old_item.update(new_item)
            new_dict[key] = old_item

        old_dict.update(new_dict)
        return old_dict


    def _merge_external_config(self):
        external_config = dict()
        external_config = self._update_dict(external_config, self.file_config)
        external_config = self._update_dict(external_config, self.variable_config)

        return external_config

    def _get_internal_config(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        init_config_path = os.path.join(current_path, "basic_config.yaml")
        internal_config = self._load_file_config(init_config_path)

        return internal_config

    def _get_final_config(self):
        final_config = dict()
        final_config = self._update_dict(final_config, self.internal_config)
        final_config = self._update_dict(final_config, self.external_config)

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
        model2path = self.final_config['model2path']
        model2pooling = self.final_config['model2pooling']
        method2index = self.final_config['method2index']

        generator_model = self.final_config['generator_model']

        if self.final_config['index_path'] is None:
            try:
                self.final_config['index_path'] = method2index[retrieval_method]
            except:
                print("Index is empty!!")
                assert False

        self.final_config['retrieval_model_path'] = model2path.get(retrieval_method, retrieval_method)
        # TODO: not support when `retrieval_model` is path

        def set_pooling_method(method, model2pooling):
            for key,value in model2pooling.items():
                if key.lower() in method.lower():
                    return value
            return 'mean'

        if self.final_config.get('retrieval_pooling_method') is None:
            self.final_config['retrieval_pooling_method'] = set_pooling_method(retrieval_method, model2pooling)


        rerank_model_name = self.final_config['rerank_model_name']
        if self.final_config.get('rerank_model_path') is None:
            if rerank_model_name is not None:
                self.final_config['rerank_model_path'] = model2path.get(rerank_model_name, rerank_model_name)
        if self.final_config['rerank_pooling_method'] is None:
            if rerank_model_name is not None:
                self.final_config['rerank_pooling_method'] = set_pooling_method(
                    rerank_model_name,
                    model2pooling
                )

        if self.final_config.get('generator_model_path') is None:
            self.final_config['generator_model_path'] = model2path.get(generator_model, generator_model)

        if 'refiner_name' in self.final_config:
            refiner_model = self.final_config['refiner_name']
            self.final_config['refiner_model_path'] = model2path.get(refiner_model, refiner_model)

        # set model path in metric setting
        metric_setting = self.final_config['metric_setting']
        metric_tokenizer_name = metric_setting.get('tokenizer_name', None)
        from flashrag.utils.constants import OPENAI_MODEL_DICT
        if metric_tokenizer_name  not in OPENAI_MODEL_DICT:
            metric_tokenizer_name = model2path.get(metric_tokenizer_name, metric_tokenizer_name)
            metric_setting['tokenizer_name'] = metric_tokenizer_name
            self.final_config['metric_setting'] = metric_setting

    def _prepare_dir(self):
        save_note = self.final_config['save_note']
        current_time = datetime.datetime.now()
        self.final_config['save_dir'] = os.path.join(self.final_config['save_dir'],
                                     f"{self.final_config['dataset_name']}_{current_time.strftime('%Y_%m_%d_%H_%M')}_{save_note}")
        os.makedirs(self.final_config['save_dir'], exist_ok=True)
        # save config parameters
        config_save_path = os.path.join(self.final_config['save_dir'],'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(self.final_config, f)

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
                "'Config' object has no attribute 'final_config'"
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