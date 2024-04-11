from flashrag.evaluator.utils import normalize_answer, is_regex
import re
from collections import Counter

class BaseMetric:
    r"""`BaseMetric` serves as the base object of all metrics. Implemented metric should
    inherit this class.
    
    """
    metric_name = "base"

    def __init__(self, config):
        self.config = config
    
    def calculate_metric(self, data):
        r"""Get the total score of this metric and score for each sample.

        Args:
            data object: it contains basic information and generated information.

        Returns:
            (metric_score: dict, metric_score_list: list)
            metric_score: such as ``{'em': 0.53}``.
            metric_score_list: score for each sample.
        
        """
        return {}, []
    
def calculate_em(prediction: str, golden_answers: list) -> float:
    if isinstance(golden_answers,str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0.0
    for golden_answer in golden_answers:
        if is_regex(golden_answer):
            print("Consider answer as regex!")
            golden_answer = re.compile(golden_answer, re.IGNORECASE)
            match = re.fullmatch(golden_answer, normalized_prediction)
            if match is not None:
                score = 1.0
                break
        else:
            golden_answer = normalize_answer(golden_answer)
            if golden_answer == normalized_prediction:
                score = 1.0
                break
    return score

def calculate_sub_em(prediction: str, golden_answers: list) -> float:
    if isinstance(golden_answers,str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0.0
    for golden_answer in golden_answers:
        if is_regex(golden_answer):
            print("Consider answer as regex!")
            golden_answer = re.compile(golden_answer, re.IGNORECASE)
            match = re.search(golden_answer, normalized_prediction)
            if match is not None:
                score = 1.0
                break
        else:
            golden_answer = normalize_answer(golden_answer)
            if golden_answer in normalized_prediction:
                score = 1.0
                break
    return score

def token_level_scores(prediction: str, ground_truths: str):
    final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
    if isinstance(ground_truths,str):
        ground_truths = [ground_truths]
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)
    
        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        for k in ['f1', 'precision', 'recall']:
            final_metric[k] = max(eval(k), final_metric[k])
    return final_metric

class Recall_Score(BaseMetric):
    """Token-level Recall score"""
    metric_name = "recall"
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = data.golden_answers
        metric_score_list = [token_level_scores(pred, golden_answers)['recall'] for pred, golden_answers in zip(pred_list, golden_answers_list)]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"recall": precision}, metric_score_list

class Precision_Score(BaseMetric):
    """Token-level Precision score"""
    metric_name = "precision"
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = data.golden_answers
        metric_score_list = [token_level_scores(pred, golden_answers)['precision'] for pred, golden_answers in zip(pred_list, golden_answers_list)]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"precision": precision}, metric_score_list

class F1_Score(BaseMetric):
    """Token-level F1 score"""
    metric_name = "f1"
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = data.golden_answers
        metric_score_list = [token_level_scores(pred, golden_answers)['f1'] for pred, golden_answers in zip(pred_list, golden_answers_list)]
        f1 = sum(metric_score_list) / len(metric_score_list)
        return {"f1": f1}, metric_score_list

class ExactMatch(BaseMetric):
    r"""Exact match measure whether the predicted answer is completely consistent 
    with the standard answer.
    
    """
    metric_name = "em"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [calculate_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)]
        em_score = sum(metric_score_list) / len(metric_score_list)
        
        return {"em": em_score}, metric_score_list
    
class Sub_ExactMatch(BaseMetric):
    r"""Sub-Exact match measure whether the predicted answer contains the standard answer.
    """
    metric_name = "sub_em"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [calculate_sub_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)]
        sub_em_score = sum(metric_score_list) / len(metric_score_list)
        
        return {"sub_em": sub_em_score}, metric_score_list
