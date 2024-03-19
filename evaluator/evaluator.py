import json
import sys
import os
import inspect
from flashrag.evaluator.metrics import BaseMetric

class Evaluator:
    r"""Evaluator is used to summarize the results of all metrics.
    
    """

    def __init__(self, config):
        self.config = config
        self.save_metric = config['save_metric_score']
        self.metrics = [metric.lower() for metric in self.config['metrics']]

        self.avaliable_metrics = self._collect_metrics()

        self.metric_class = {}
        for metric in self.metrics:
            if metric in self.avaliable_metrics:
                self.metric_class[metric] = self.avaliable_metrics[metric](self.config)
            else:
                print(f"{metric} has not been implemented!")
                raise NotImplementedError
        
        self.save_metric_score = config['save_metric_score']
    
    def _collect_metrics(self):
        f"""Collect all classes based on ```BaseMetric```.
        
        """
        avaliable_metrics = {}

        for cls in BaseMetric.__subclasses__():
            metric_name = cls.metric_name
            avaliable_metrics[metric_name] = cls

        return avaliable_metrics

    def evaluate(self, data):
        f"""Calculate all metric indicators and summarize them.

        """

        result_dict = {}
        for metric in self.metrics:
            metric_result, metric_scores = self.metric_class[metric].calculate_metric(data)
            result_dict.update(metric_result)
            
            for metric_score, item in zip(metric_scores, data):
                item.update_evaluation_score(metric, metric_score)
            
            if self.save_metric_score:
                self.save(data)

        
        return result_dict
    
    def save(self, data):
        r"""Save the evaluated data, including the raw data and the score of each data 
        sample on each metric.
        
        """
        
        save_dir = self.config['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{data.dataset_name}_evaluated.jsonl"
        save_path = os.path.join(save_dir, file_name)

        data.save(save_path)
