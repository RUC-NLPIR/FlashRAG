import json
import os
import random
import numpy as np

class Item:
    r"""A container class used to store and manipulate a sample within a dataset. 
    Information related to this sample during training/inference will be stored in ```self.output```.
    Each attribute of this class can be used like a dict key(also for key in ```self.output```).
    
    """
    # __setitem__ = dict.__setattr__
    # __getitem__ = dict.__getattribute__

    def __init__(self, item_dict):
        self.id = item_dict.get("id")
        self.question = item_dict.get("question")
        self.golden_answers = item_dict.get("golden_answers")
        self.metadata = item_dict.get("metadata",{})
        self.output = item_dict.get("output", {})

    def update_output(self, key, value):
        r"""Update the output dict and keep a key in self.output can be used as an attribute.
        
        """
        if key in ['id','question','golden_answers','output']:
            raise AttributeError(f'{key} should not be changed')
        else:
            self.output[key] = value

    def update_evaluation_score(self, metric_name, metric_score):
        r"""Update the evaluation score of this sample for a metric.
        
        """
        if 'metric_score' not in self.output:
            self.output['metric_score'] = {}
        self.output['metric_score'][metric_name] = metric_score
    
    # def __getitem__(self, key):
    #     if key in ['id','question','golden_answers','metadata']:
    #         return getattr(self, key)
    #     else:
    #         return self.output[key]
    
    # def __setitem__(self, key, value):
    #     if key in ['id','question','golden_answers','metadata']:
    #         setattr(self, key)
    #     else:
    #         self.output[key] = value

    def __getattr__(self, attr_name):
        if attr_name in ['id','question','golden_answers','metadata','output']:
            return super().__getattribute__(attr_name)
        else:
            output = super().__getattribute__('output')
            if attr_name in output:
                return output[attr_name]
            else:
                raise AttributeError(f"Attribute `{attr_name}` not found")

    def to_dict(self):
        r"""Convert all information within the data sample into a dict. Information generated
        during the inference will be saved into output field.
        
        """
        for k,v in self.output.items():
            if isinstance(k, np.ndarray):
                self.output[k] = v.tolist()
        output =  {
            "id": self.id,
            "question": self.question,
            "golden_answers": self.golden_answers,
            "output": self.output
        }
        if self.metadata != {}:
            output['metadata'] = self.metadata

        return output
    

class Dataset:
    r"""A container class used to store the whole dataset. Inside the class, each data sample will be stored
    in ```Item``` class.
    The properties of the dataset represent the list of attributes corresponding to each item in the dataset.
    
    """
    
    def __init__(self, config=None, dataset_path=None, data=None, sample_num = None, random_sample = False):
        self.config = config
        self.dataset_name = config['dataset_name']
        self.dataset_path = dataset_path

        self.sample_num = sample_num
        self.random_sample = random_sample

        if data is None:
            self.data = self._load_data(self.dataset_name, self.dataset_path)
        else:
            self.data = data

    def _load_data(self, dataset_name, dataset_path):
        r"""Load data from the provided dataset_path or directly download the file(TODO). 
        
        """

        if not os.path.exists(dataset_path):
            # TODO: auto download: self._download(dataset_name, dataset_path)
            pass

        data = []
        with open(dataset_path,"r",encoding="utf-8") as f:
            for line in f:
                item_dict = json.loads(line)
                item = Item(item_dict)
                data.append(item)
        if self.sample_num is not None:
            if self.random_sample:
                print(f"Random sample {self.sample_num} items in test set.")
                data = random.sample(data, self.sample_num)
            else:
                data = data[:self.sample_num]

        return data
    
    def update_output(self, key, value_list):
        r"""Update the overall output field for each sample in the dataset.
        
        """

        assert len(self.data) == len(value_list)
        for item, value in zip(self.data, value_list):
            item.update_output(key, value)

    @property
    def question(self):
        return [item.question for item in self.data]
    @property
    def golden_answers(self):
        return [item.golden_answers for item in self.data]
    @property
    def id(self):
        return [item.id for item in self.data]
    @property
    def output(self):
        return [item.output for item in self.data]

    def get_batch_data(self, attr_name:str, batch_size: int):
        r"""Get an attribute of dataset items in batch.
        
        """

        for i in range(0, len(self.data), batch_size):
            batch_items = self.data[i:i+batch_size]
            yield [item[attr_name] for item in batch_items]
    
    def __getattr__(self, attr_name):
        return [item.__getattr__(attr_name) for item in self.data]


    def get_attr_data(self, attr_name):
        r"""For the attributes constructed later (not implemented using property), 
        obtain a list of this attribute in the entire dataset. 
        
        """
        return [item[attr_name] for item in self.data]
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def save(self, save_path):
        f"""Save the dataset into the original format.

        """
        save_data = [item.to_dict() for item in self.data]
        with open(save_path,"w") as f:
            json.dump(save_data, f, indent=4)
            # for item in save_data:
            #     json.dump(item, f)
            #     f.write("\n")

        




        
        