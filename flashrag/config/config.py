import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import random
import datetime


class Config:
    def __init__(self, config_file_path=None, config_dict={}):

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
        if not self.final_config.get('disable_save', False):
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

    def _load_file_config(self, config_file_path: str):
        file_config = dict()
        if config_file_path:
            with open(config_file_path, "r", encoding="utf-8") as f:
                file_config.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config

    @staticmethod
    def _update_dict(old_dict: dict, new_dict: dict):
        # Update the original update method of the dictionary:
        # If there is the same key in `old_dict` and `new_dict`, and value is of type dict, update the key in dict

        same_keys = []
        for key, value in new_dict.items():
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
        split = self.final_config["split"]
        if split is None:
            split = ["train", "dev", "test"]
        if isinstance(split, str):
            split = [split]
        self.final_config["split"] = split

    def _init_device(self):
        gpu_id = self.final_config["gpu_id"]
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            # import pynvml 
            # pynvml.nvmlInit()
            # gpu_num = pynvml.nvmlDeviceGetCount()
            import torch
            gpu_num = torch.cuda.device_count()
        except:
            gpu_num = 0
        self.final_config['gpu_num'] = gpu_num
        if gpu_num > 0:
            self.final_config["device"] = "cuda"
        else:
            self.final_config['device'] = 'cpu'

    def _set_additional_key(self):
        def set_pooling_method(method, model2pooling):
            for key, value in model2pooling.items():
                if key.lower() in method.lower():
                    return value
            return "mean"

        def set_retrieval_keys(model2path, model2pooling, method2index, config):
            retrieval_method = config["retrieval_method"]
            if config["index_path"] is None:
                try:
                    config["index_path"] = method2index[retrieval_method]
                except:
                    print("Index is empty!!")

            if config.get("retrieval_model_path") is None:
                config["retrieval_model_path"] = model2path.get(retrieval_method, retrieval_method)

            if config.get("retrieval_pooling_method") is None:
                config["retrieval_pooling_method"] = set_pooling_method(retrieval_method, model2pooling)

            rerank_model_name = config.get("rerank_model_name", None)
            if config.get("rerank_model_path", None) is None:
                if rerank_model_name is not None:
                    config["rerank_model_path"] = model2path.get(rerank_model_name, rerank_model_name)
            if config.get("rerank_pooling_method", None) is None:
                if rerank_model_name is not None:
                    config["rerank_pooling_method"] = set_pooling_method(rerank_model_name, model2pooling)
            return config

        # set dataset
        dataset_name = self.final_config["dataset_name"]
        data_dir = self.final_config["data_dir"]
        self.final_config["dataset_path"] = os.path.join(data_dir, dataset_name)

        # set retrieval-related keys
        model2path = self.final_config["model2path"]
        model2pooling = self.final_config["model2pooling"]
        method2index = self.final_config["method2index"]
        self.final_config = set_retrieval_keys(model2path, model2pooling, method2index, self.final_config)
        # set keys for multi retriever
        if "multi_retriever_setting" in self.final_config:
            multi_retriever_config = self.final_config["multi_retriever_setting"]
            retriever_config_list = multi_retriever_config.get("retriever_list", [])
            # set for reranker merge method
            assert multi_retriever_config['merge_method'] in ['concat', 'rrf', 'rerank', None]
            if multi_retriever_config['merge_method'] == 'rerank':
                rerank_model_name = multi_retriever_config.get("rerank_model_name", None)
                assert rerank_model_name is not None
                multi_retriever_config['rerank_max_length'] = multi_retriever_config.get("rerank_max_length", 512)
                multi_retriever_config['rerank_batch_size'] = multi_retriever_config.get("rerank_batch_size", 256)
                multi_retriever_config['rerank_use_fp16'] = multi_retriever_config.get("rerank_use_fp16", True)
                
                if multi_retriever_config.get("rerank_model_path", None) is None:
                    if rerank_model_name is not None:
                        multi_retriever_config["rerank_model_path"] = model2path.get(rerank_model_name, rerank_model_name)
                if multi_retriever_config.get("rerank_pooling_method", None) is None:
                    if rerank_model_name is not None:
                        multi_retriever_config["rerank_pooling_method"] = set_pooling_method(rerank_model_name, model2pooling)
            
            # set config for each retriever
            for retriever_config in retriever_config_list:
                if "instruction" not in retriever_config:
                    retriever_config["instruction"] = None
                if "bm25_backend" not in retriever_config:
                    retriever_config["bm25_backend"] = "bm25s"
                if "use_reranker" not in retriever_config:
                    retriever_config["use_reranker"] = False
                if "index_path" not in retriever_config:
                    retriever_config["index_path"] = None
                if "corpus_path" not in retriever_config:
                    retriever_config["corpus_path"] = None
                if "use_sentence_transformer" not in retriever_config:
                    retriever_config["use_sentence_transformer"] = False
                retriever_config = set_retrieval_keys(model2path, model2pooling, method2index, retriever_config)
                
                # set other necessary keys as base setting
                keys = [
                    "retrieval_use_fp16",
                    "retrieval_query_max_length",
                    "faiss_gpu",
                    "retrieval_topk",
                    "retrieval_batch_size",
                    "use_reranker",
                    "rerank_model_name",
                    "rerank_model_path",
                    "retrieval_cache_path",
                ]
                for key in keys:
                    if key not in retriever_config:
                        retriever_config[key] = self.final_config.get(key, None)
                retriever_config["save_retrieval_cache"] = False
                retriever_config["use_retrieval_cache"] = False
        
        # set model path
        generator_model = self.final_config["generator_model"]

        if self.final_config.get("generator_model_path") is None:
            self.final_config["generator_model_path"] = model2path.get(generator_model, generator_model)

        if "refiner_name" in self.final_config:
            refiner_model = self.final_config["refiner_name"]
            if "refiner_model_path" not in self.final_config or self.final_config["refiner_model_path"] is None:
                self.final_config["refiner_model_path"] = model2path.get(refiner_model, None)
        if "instruction" not in self.final_config:
            self.final_config["instruction"] = None

        # set model path in metric setting
        metric_setting = self.final_config["metric_setting"]
        metric_tokenizer_name = metric_setting.get("tokenizer_name", None)
        from flashrag.utils.constants import OPENAI_MODEL_DICT

        if metric_tokenizer_name not in OPENAI_MODEL_DICT:
            metric_tokenizer_name = model2path.get(metric_tokenizer_name, metric_tokenizer_name)
            metric_setting["tokenizer_name"] = metric_tokenizer_name
            self.final_config["metric_setting"] = metric_setting

    def _prepare_dir(self):
        save_note = self.final_config["save_note"]
        save_dir = self.final_config['save_dir']
        if not save_dir.endswith("/"):
            save_dir += "/"

        current_time = datetime.datetime.now()

        self.final_config["save_dir"] = os.path.join(
            save_dir,
            f"{self.final_config['dataset_name']}_{current_time.strftime('%Y_%m_%d_%H_%M')}_{save_note}",
        )
        os.makedirs(self.final_config["save_dir"], exist_ok=True)
        # save config parameters
        config_save_path = os.path.join(self.final_config["save_dir"], "config.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(self.final_config, f, indent=4, sort_keys=False)

    def _set_seed(self):
        import torch
        import numpy as np
        seed = self.final_config['seed']
        try:
            seed = int(seed)
        except:
            seed = 2025
        self.final_config['seed'] = seed
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
            raise AttributeError("'Config' object has no attribute 'final_config'")
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
