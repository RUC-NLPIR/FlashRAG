from flashrag.pipeline import SequentialPipeline
from flashrag.dataset import Dataset, Item
from chat_pipelines.base_chat_pipeline import BaseChatPipeline
from flashrag.pipeline import IterativePipeline, SelfRAGPipeline, FLAREPipeline, SelfAskPipeline, IRCOTPipeline
import re

class IterativePipeline_Chat(BaseChatPipeline,IterativePipeline):
    def __init__(
        self,
        config,
        prompt_template=None,
        retriever=None,
        generator=None
    ):
        iter_num = config.get('iter_num', 3)
        BaseChatPipeline.__init__(self, config)
        IterativePipeline.__init__(
            self, 
            config = config,
            prompt_template = prompt_template,
            retriever = retriever,
            generator = generator,
            iter_num = iter_num
        )
    
    def chat(self, query):
        dataset = Dataset(config = self.config, data = [Item({"question": query})])

        questions = dataset.question

        past_generation_result = []  # list of N items
        for iter_idx in range(self.iter_num):
            if iter_idx == 0:
                input_query = questions
            else:
                assert len(questions) == len(past_generation_result)
                input_query = [f"{q} {r}" for q, r in zip(questions, past_generation_result)]

            # generation-augmented retrieval
            retrieval_results = self.retriever.batch_search(input_query)
            dataset.update_output(f"retrieval_result_iter_{iter_idx}", retrieval_results)

            yield from self.display_middle_result('', f"retrieval_result_iter_{iter_idx}")
            yield from self.display_retrieval_result(retrieval_results[0])

            # retrieval-augmented generation
            # input_prompts = self.build_prompt(questions, retrieval_results)
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(questions, retrieval_results)
            ]

            dataset.update_output(f"prompt_iter_{iter_idx}", input_prompts)
            past_generation_result = self.generator.generate(input_prompts)
            dataset.update_output(f"pred_iter_{iter_idx}", past_generation_result)
            yield from self.display_middle_result(past_generation_result, f"pred_iter_{iter_idx}")

class SelfRAGPipeline_Chat(BaseChatPipeline, SelfRAGPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        selfrag_settings = config['self_rag_setting']
        selfrag_settings['config'] = config
        selfrag_settings['prompt_template'] = prompt_template
        selfrag_settings['retriever'] = retriever
        selfrag_settings['generator'] = generator
        BaseChatPipeline.__init__(self, config)
        SelfRAGPipeline.__init__(self, **selfrag_settings)
    
    def chat(self, query):
        dataset = Dataset(config = self.config, data = [Item({"question": query})])
        
        item = next(dataset)

        question = item.question
        retrieval_result, scores = self.retriever.search(question, return_score=True)
        yield from self.display_retrieval_result(retrieval_result)

        doc2score = {doc_item["id"]: score for doc_item, score in zip(retrieval_result, scores)}
        id2doc = {doc_item["id"]: doc_item for doc_item in retrieval_result}

        thoughts = []
        iter_num = 0
        while iter_num < self.max_iter:
            input_prompt = self.prompt_template.get_string(
                question=question, retrieval_result=retrieval_result, previous_gen=" ".join(thoughts)
            )
            yield from self.display_middel_result('', f'Intermediate_output_iter{iter_num}')
            yield from self.display_middle_result(input_prompt, 'Middle input prompt')
            new_thought = self.generator.generate(input_prompt,stop=['.', '\n'])[0]
            yield from self.display_middle_result(new_thought, 'New thought')

            thoughts.append(new_thought)
            iter_num += 1
            if "So the answer is:" in new_thought:
                item.update_output(
                    f"intermediate_output_iter{iter_num}",
                    {
                        "input_prompt": input_prompt,
                        "new_thought": new_thought,
                    },
                )
                break

            # retrieve new docs and merge
            new_retrieval_result, new_scores = self.retriever.search(new_thought, return_score=True)
            for doc_item, score in zip(new_retrieval_result, new_scores):
                id2doc[doc_item["id"]] = doc_item
                doc_id = doc_item["id"]
                if doc_id in doc2score:
                    doc2score[doc_id] = max(doc2score[doc_id], score)
                else:
                    doc2score[doc_id] = score
            sorted_doc_score = sorted(doc2score.items(), key=lambda x: x[1], reverse=False)
            sorted_doc_id = [t[0] for t in sorted_doc_score]
            retrieval_result = [id2doc[id] for id in sorted_doc_id]

            yield from self.display_retrieval_result(retrieval_result)

            item.update_output(
                f"intermediate_output_iter{iter_num}",
                {
                    "input_prompt": input_prompt,
                    "new_thought": new_thought,
                    "new_retreival_result": new_retrieval_result,
                },
            )

        self.display_middle_result(" ".join(thoughts), 'Final Answer')


class FLAREPipeline_Chat(BaseChatPipeline, FLAREPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        flare_settings = config['Flare']
        flare_settings['config'] = config
        flare_settings['prompt_template'] = prompt_template
        flare_settings['retriever'] = retriever
        flare_settings['generator'] = generator
        BaseChatPipeline.__init__(self, config)
        FLAREPipeline.__init__(self, **flare_settings)

    def chat(self, query):
        dataset = Dataset(config = self.config, data = [Item({"question": query})])
        item = next(dataset)

        question = item.question
        gen_length = 0
        iter_round = 0
        final_gen_result = ""
        while gen_length < self.max_generation_length and iter_round < self.max_iter_num:
            input_prompt = self.prompt_template.get_string(question=question, previous_gen=final_gen_result)
            # input_prompt = self.build_prompt(
            #     question_list=[question], use_reference=False, previous_gen=final_gen_result)[0]
            # scores: token logits of the whole generation seq
            round_gen_output, scores = self.generator.generate(
                input_prompt, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps
            )
            round_gen_output, scores = round_gen_output[0], scores[0]
            # next_sent_scores: token logits of the first sent in generation seq
            next_sent, next_sent_score = self.get_next_sentence(round_gen_output, scores)

            yield from self.display_middle_result(next_sent, 'Next Sentence')

            # judge next sentence
            judge_result, query = self.judge_sent_confidence(next_sent, next_sent_score)
            item.update_output(f"judge_result_iter{iter_round}", judge_result)

            yield from self.display_middle_result(str(judge_result), 'Judge Result')

            if not judge_result:
                # do retrieval-augmented generation
                retrieval_result = self.retriever.search(query)
                item.update_output("retrieval_result", retrieval_result)
                input_prompt = self.prompt_template.get_string(
                    question=question, retrieval_result=retrieval_result, previous_gen=final_gen_result
                )

                # input_prompt = self.build_prompt(
                #     question_list = [question],
                #     retrieval_results = [retrieval_result],
                #     previous_gen = final_gen_result)[0]
                output, scores = self.generator.generate(
                    input_prompt, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps
                )
                output, scores = output[0], scores[0]
                next_sent, _ = self.get_next_sentence(output, scores)
                item.update_output(f"gen_iter_{iter_round}", next_sent)
                item.update_output("retrieval_result", retrieval_result)

            final_gen_result += next_sent

            yield from self.display_middle_result(final_gen_result, 'Current Generation Result')
            gen_length += len(next_sent_score)
            iter_round += 1

class SelfAskPipeline_Chat(BaseChatPipeline, SelfAskPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        settings = config['Ret-Robust']
        settings['config'] = config
        settings['prompt_template'] = prompt_template
        settings['retriever'] = retriever
        settings['generator'] = generator
        BaseChatPipeline.__init__(self, config)
        SelfAskPipeline.__init__(self, **settings)

    def chat(self, query):
        dataset = Dataset(config = self.config, data = [Item({"question": query})])
        item = next(dataset)

        question = item.question
        retrieval_result = self.retriever.search(question)
        yield from self.display_retrieval_result(retrieval_result)

        stop_condition = "Intermediate answer:"
        follow_ups = "No." if self.single_hop else "Yes."
        res = ""
        early_exit = False
        for idx in range(self.max_iter):
            input_prompt = (
                self.P_INS
                + "\n"
                + self.format_reference(retrieval_result)
                + f"\nQuesiton: {question}"
                + "\nAre follow up questions needed here: "
                + follow_ups
                + "\n"
                + res
            )
            gen_out = self.generator.generate(input_prompt, stop=["Context:", "#", stop_condition])[0]
            item.update_output(f"intermediate_output_iter{idx}", gen_out)
            yield from self.display_middle_result(gen_out, f"intermediate_output_iter{idx}")

            if stop_condition == "Intermediate answer:":
                res += gen_out.split("Intermediate answer:")[0]
                stop_condition = "Follow up:"

            elif stop_condition == "Follow up:":
                followup_split = re.split(self.FOLLOW_UP_PATTERN, gen_out)
                res += followup_split[0]

                if len(followup_split) > 1:
                    res += re.findall(self.FOLLOW_UP_PATTERN, gen_out)[0]
                stop_condition = "Intermediate answer:"

            # make sure the result does not end in a new line
            if len(res) == 0:
                early_exit = True
                break
            if res[-1] == "\n":
                res = res[:-1]

            if "Follow up: " in gen_out:
                # get the first follow up
                new_query = [l for l in gen_out.split("\n") if "Follow up: " in l][0].split("Follow up: ")[-1]
                retrieval_result = self.retriever.search(new_query)
                yield from self.display_retrieval_result(retrieval_result)

            if "So the final answer is: " in gen_out:
                res = (
                    self.format_reference(retrieval_result)
                    + f"\nQuesiton: {question}"
                    + "\nAre follow up questions needed here: "
                    + follow_ups
                    + "\n"
                    + res
                )
                early_exit = True
                # print("Success: early exit!")
                break

        if not early_exit:
            res = (
                self.format_reference(retrieval_result)
                + f"\nQuesiton: {question}"
                + "\nAre follow up questions needed here: "
                + follow_ups
                + "\n"
                + res
            )

        item.update_output("retrieval_result", retrieval_result)
        item.update_output("pred", res)
        yield from self.display_middle_result(res, 'Final Answer')


class IRCOTPipeline_Chat(BaseChatPipeline, IRCOTPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        settings = config['IRCOT'] if config['IRCOT'] is not None else {'max_iter': 5}
        settings['config'] = config
        settings['prompt_template'] = prompt_template
        settings['retriever'] = retriever
        settings['generator'] = generator
        BaseChatPipeline.__init__(self, config)
        IRCOTPipeline.__init__(self, **settings)
    
    def chat(self, query):
        dataset = Dataset(config = self.config, data = [Item({"question": query})])
        item = dataset.data[0]

        question = item.question
        retrieval_result, scores = self.retriever.search(question, return_score=True)
        yield from self.display_retrieval_result(retrieval_result)

        doc2score = {doc_item["id"]: score for doc_item, score in zip(retrieval_result, scores)}
        id2doc = {doc_item["id"]: doc_item for doc_item in retrieval_result}

        thoughts = []
        iter_num = 0
        while iter_num < self.max_iter:
            input_prompt = self.prompt_template.get_string(
                question=question, retrieval_result=retrieval_result, previous_gen=" ".join(thoughts)
            )
            new_thought = self.generator.generate(input_prompt,stop=['.', '\n'])[0]

            yield from self.display_middle_result(new_thought, 'Middle Generation Result')

            thoughts.append(new_thought)
            iter_num += 1
            if "So the answer is:" in new_thought:
                item.update_output(
                    f"intermediate_output_iter{iter_num}",
                    {
                        "input_prompt": input_prompt,
                        "new_thought": new_thought,
                    },
                )
                break

            # retrieve new docs and merge
            new_retrieval_result, new_scores = self.retriever.search(new_thought, return_score=True)
            for doc_item, score in zip(new_retrieval_result, new_scores):
                id2doc[doc_item["id"]] = doc_item
                doc_id = doc_item["id"]
                if doc_id in doc2score:
                    doc2score[doc_id] = max(doc2score[doc_id], score)
                else:
                    doc2score[doc_id] = score
            sorted_doc_score = sorted(doc2score.items(), key=lambda x: x[1], reverse=False)
            sorted_doc_id = [t[0] for t in sorted_doc_score]
            retrieval_result = [id2doc[id] for id in sorted_doc_id]

            yield from self.display_retrieval_result(retrieval_result)

            item.update_output(
                f"intermediate_output_iter{iter_num}",
                {
                    "input_prompt": input_prompt,
                    "new_thought": new_thought,
                    "new_retreival_result": new_retrieval_result,
                },
            )

        item.update_output("retrieval_result", retrieval_result)
        item.update_output("pred", " ".join(thoughts))
        yield from self.display_middle_result(" ".join(thoughts), 'Final Answer')
