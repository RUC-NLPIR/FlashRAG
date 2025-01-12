from flashrag.pipeline import REPLUGPipeline, SuRePipeline
from flashrag.dataset import Dataset, Item
from flashrag.utils import get_judger
from flashrag.prompt import PromptTemplate
from transformers import LogitsProcessorList
from chat_pipelines.base_chat_pipeline import BaseChatPipeline
import numpy as np
import itertools
from typing import List
import re

class REPLUGPipeline_Chat(BaseChatPipeline, REPLUGPipeline):
    def __init__(self, *args, **kwargs):
        REPLUGPipeline.__init__(self, *args, **kwargs)
        config = self.config
        BaseChatPipeline.__init__(self, config)
    
    def chat(self, query):
        dataset = Dataset(
            config = self.config,
            data = [Item({"question": query})]
        )

        import torch
        from flashrag.pipeline.replug_utils import REPLUGLogitsProcessor

        input_query = dataset.question
        retrieval_results, doc_scores = self.retriever.batch_search(
            input_query, return_score=True
        )
        
        yield from self.display_middle_result(retrieval_results, 'Retrieval result')

        pred_answer_list = []
        # each doc has a prompt
        for item in dataset:
            docs = [self.format_reference(doc_item) for doc_item in item.retrieval_result]
            prompts = self.build_single_doc_prompt(
                question=item.question, doc_list=docs
            )

            scores = torch.tensor(
                item.doc_scores, dtype=torch.float32
            ).to(self.device)
            
            output = self.generator.generate(
                prompts, batch_size=len(docs), logits_processor=LogitsProcessorList([REPLUGLogitsProcessor(scores)])
            )
            
            # the output of the batch is same
            output = output[0]
            pred_answer_list.append(output)

        dataset.update_output("pred", pred_answer_list)
        yield from self.display_middle_result(pred_answer_list, 'Final Answer')
      

class SuRePipeline_Chat(BaseChatPipeline, SuRePipeline):
    def __init__(self, *args, **kwargs):
        SuRePipeline.__init__(self, *args, **kwargs)
        config = self.config
        BaseChatPipeline.__init__(self, config)
    
    def chat(self, query):
        dataset = Dataset(config = self.config, data = [Item({"question": query})])
        input_query = dataset.question

        retrieval_results, doc_scores = self.retriever.batch_search(input_query, return_score=True)
        dataset.update_output("retrieval_result", retrieval_results)
        yield from self.display_middle_result(retrieval_results, 'Retrieval result')

        pred_answer_list = []
        for item in dataset:
            retrieval_result = item.retrieval_result
            doc_num = len(retrieval_result)
            # format all docs
            for doc_item in retrieval_result:
                if "title" not in doc_item or "text" not in doc_item:
                    doc_item["title"] = doc_item["contents"].split("\n")[0]
                    doc_item["text"] = "\n".join(doc_item["contents"].split("\n")[1:])
            formatted_ref = self.format_ref(
                titles=[i["title"] for i in retrieval_result], texts=[i["text"] for i in retrieval_result]
            )
            
            # get candidates
            input_prompt = self.P_CAN_TEMPLATE.get_string(
                N = doc_num,
                formatted_reference = formatted_ref,
                question = item.question
            )
            yield from self.display_middle_result(input_prompt, 'Candidate Generation')
            
            output = self.generator.generate([input_prompt])[0]
            candidates = self.parse_candidates(output)
            item.update_output("candidates", candidates)
            yield from self.display_middle_result(str(candidates), 'Candidate Generation Result')

            if len(candidates) == 0:
                print("No valid predictions!")
                pred = ""
                pred_answer_list.append(pred)
                continue

            # get summarization for each candidate
            input_prompts = [
                self.P_SUM_TEMPLATE.get_string(question=item.question, pred=cand, formatted_reference=formatted_ref)
                for cand in candidates
            ]

            all_summary = self.generator.generate(input_prompts)
            item.update_output("all_summary", all_summary)
            yield from self.display_middle_result(str(all_summary), 'Candidate Summary Result')


            # instance-wise validation
            input_prompts = [
                self.P_VAL_TEMPLATE.get_string(
                    question = item.question,
                    pred = cand,
                    summary = summary
                )
                for cand, summary in zip(candidates, all_summary)
            ]
            val_results = self.generator.generate(input_prompts)
            val_scores = [self.parse_validation(res) for res in val_results]
            item.update_output("val_scores", val_scores)
            yield from self.display_middle_result(str(val_scores), 'Val Scores')


            # pair-wise ranking
            summary_num = len(all_summary)
            score_matrix = np.zeros((summary_num, summary_num))
            iter_idxs = list(itertools.permutations(range(summary_num), 2))
            input_prompts = [
                self.P_RANK_TEMPLATE.get_string(
                    question=item.question, summary1=all_summary[idx_tuple[0]], summary2=all_summary[idx_tuple[1]]
                )
                for idx_tuple in iter_idxs
            ]
            ranking_output = self.generator.generate(input_prompts)
            ranking_scores = [self.parse_ranking(res) for res in ranking_output]
            for idx_tuple, score in zip(iter_idxs, ranking_scores):
                score_matrix[idx_tuple[0], idx_tuple[1]] = score
            ranking_scores = score_matrix.sum(axis=1).squeeze().tolist()  # ranking score for each summary
            item.update_output("ranking_scores", ranking_scores)
            yield from self.display_middle_result(str(ranking_scores), 'Ranking Scores')


            # combine two scores as the final score for each summary
            if not isinstance(ranking_scores, list):
                ranking_scores = [ranking_scores]
            if not isinstance(val_scores, list):
                val_scores = [val_scores]
            total_scores = [x + y for x, y in zip(val_scores, ranking_scores)]

            best_idx = np.argmax(total_scores)
            pred = candidates[best_idx]
            pred_answer_list.append(pred)
            yield from self.display_middle_result(pred, 'Final Answer')