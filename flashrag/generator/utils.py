# import torch
# from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
#     STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings

from transformers import StoppingCriteria
from torch import LongTensor, FloatTensor, eq



class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if eq(input_ids[0][-len(stop_ids[0])+1:], stop_ids[0][1:]).all():
                return True
        return False



# class StopAtSpecificTokenCriteria(StoppingCriteria):
#     def __init__(self, token_id_list = None):
#         self.token_id_list = token_id_list
        
#     @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list