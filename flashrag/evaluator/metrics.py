from flashrag.evaluator.utils import normalize_answer, is_regex
import re

class BaseMetric:
    r"""`BaseMetric` serves as the base object of all metrics. Implemented metric should
    inherit this class.
    
    """
    metric_name = "base"

    def __init__(self, config):
        self.config = config
    
    def calculate_metric(self, data, save_metric = True):
        r"""Get the total score of this metric and score for each sample.

        Args:
            data object: it contains basic information and generated information.
            save_metric: if set to True, origin data with metric score will be saved.

        Returns:
            (metric_score: dict, metric_score_list: list)
            metric_score: such as ``{'em': 0.53}``.
            metric_score_list: score for each sample.
        
        """
        return {}, []
    


class ExactMatch(BaseMetric):
    r"""Exact match measure whether the predicted answer is completely consistent 
    with the standard answer.
    
    """
    
    metric_name = "em"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data):
        def calculate_em(prediction: str, golden_answers: list) -> float:
            if isinstance(golden_answers,str):
                golden_answers = [golden_answers]
            normalized_prediction = normalize_answer(prediction)
            score = 0.0
            for golden_answer in golden_answers:
                if is_regex(golden_answer):
                    golden_answer = re.compile(golden_answer, re.IGNORECASE)
                    match = re.fullmatch(golden_answer, normalized_prediction.lower())
                    if match is not None:
                        score = 1.0
                        break
                else:
                    golden_answer = normalize_answer(golden_answer)
                    if golden_answer.lower() == normalized_prediction.lower():
                        score = 1.0
                        break
            return score
        
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [calculate_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)]
        em_score = sum(metric_score_list) / len(metric_score_list)
        

        return {"em": em_score}, metric_score_list
    
