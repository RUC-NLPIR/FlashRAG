import os
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from flashrag.refiner import BaseRefiner
from flashrag.prompt import PromptTemplate
from flashrag.retriever.encoder import Encoder, STEncoder
from flashrag.utils import hash_object


class KGTraceRefiner(BaseRefiner):
    def __init__(self, config, retriever=None, generator=None):
        super().__init__(config)
        self.config = config
        self.input_prompt_flag = False

        default_setting = {
            "num_examplars": 3,
            "max_chain_length": 4,
            "topk_triple_select": 5,  # num of candidate triples
            "num_choices": 20,
            "min_triple_prob": 1e-4,
            "num_beams": 5,  # number of selected prob at each step of constructing chain
            "num_chains": 20,  # number of generated chains
            "n_context": 5,  # number of used chains in generation
            "context_type": "triples",  # triples/triple-doc
            "triple_save_path": os.path.join(config["save_dir"], "save_triples.json"),
            "triple_load_path": None,
        }
        if "trace_config" in config and config["trace_config"] is not None:
            default_setting.update(config["trace_config"])
        self.kg_setting = default_setting

        self.num_examplars = self.kg_setting["num_examplars"]
        self.max_chain_length = self.kg_setting["max_chain_length"]
        self.topk_triple_select = self.kg_setting["topk_triple_select"]
        self.num_beams = self.kg_setting["num_beams"]
        self.num_chains = self.kg_setting["num_chains"]
        self.num_choices = self.kg_setting["num_choices"]
        self.min_triple_prob = self.kg_setting["min_triple_prob"]
        self.n_context = self.kg_setting["n_context"]
        self.context_type = self.kg_setting["context_type"]
        self.triple_save_path = self.kg_setting["triple_save_path"]
        self.triple_load_path = self.kg_setting["triple_load_path"]

        # set necessary component
        if retriever is None:
            print("Load new retriever")
            from flashrag.utils import get_retriever

            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever
        if generator is None:
            print("Load new generator")
            from flashrag.utils import get_generator

            self.generator = get_generator(config)
        else:
            self.generator = generator

        # load demonstrations
        if config["retrieval_method"] != "e5":
            self.encoder = Encoder(
                model_name="e5",
                model_path=config["model2path"]["e5"],
                pooling_method="mean",
                max_length=256,
                use_fp16=True,
            )
        else:
            self.encoder = self.retriever.encoder

        # load demonstrations for generating triples and reasoning chain
        if config["dataset_name"].lower() == "hotpotqa":
            from flashrag.prompt.trace_examplars import (
                TRIPLE_EXAMPLARS_HOTPOTQA,
                GENERATING_CHAIN_EXAMPLARS_HOTPOTQA,
                FINAL_CHAIN_EXAMPLARS_HOTPOTQA,
            )

            self.triple_examplars = TRIPLE_EXAMPLARS_HOTPOTQA
            self.final_chain_examplars = FINAL_CHAIN_EXAMPLARS_HOTPOTQA
            self.generating_chain_examplars = GENERATING_CHAIN_EXAMPLARS_HOTPOTQA
        elif config["dataset_name"].lower() == "musique":
            from flashrag.prompt.trace_examplars import (
                TRIPLE_EXAMPLARS_MUSIQUE,
                GENERATING_CHAIN_EXAMPLARS_MUSIQUE,
                FINAL_CHAIN_EXAMPLARS_MUSIQUE,
            )

            self.triple_examplars = TRIPLE_EXAMPLARS_MUSIQUE
            self.final_chain_examplars = FINAL_CHAIN_EXAMPLARS_MUSIQUE
            self.generating_chain_examplars = GENERATING_CHAIN_EXAMPLARS_MUSIQUE
        elif "2wiki" in config["dataset_name"].lower():
            from flashrag.prompt.trace_examplars import (
                TRIPLE_EXAMPLARS_2WIKIMULTIHOPQA,
                GENERATING_CHAIN_EXAMPLARS_2WIKIMULTIHOPQA,
                FINAL_CHAIN_EXAMPLARS_2WIKIMULTIHOPQA,
            )

            self.triple_examplars = TRIPLE_EXAMPLARS_2WIKIMULTIHOPQA
            self.final_chain_examplars = FINAL_CHAIN_EXAMPLARS_2WIKIMULTIHOPQA
            self.generating_chain_examplars = GENERATING_CHAIN_EXAMPLARS_2WIKIMULTIHOPQA
        else:
            # use hotpotqa examplars
            from flashrag.prompt.trace_examplars import (
                TRIPLE_EXAMPLARS_HOTPOTQA,
                GENERATING_CHAIN_EXAMPLARS_HOTPOTQA,
                FINAL_CHAIN_EXAMPLARS_HOTPOTQA,
            )

            self.triple_examplars = TRIPLE_EXAMPLARS_HOTPOTQA
            self.final_chain_examplars = FINAL_CHAIN_EXAMPLARS_HOTPOTQA
            self.generating_chain_examplars = GENERATING_CHAIN_EXAMPLARS_HOTPOTQA

        triple_examplars_text_list = [f"title: {item['title']} text: {item['text']}" for item in self.triple_examplars]
        self.triple_examplars_embeddings = self.encoder.encode(triple_examplars_text_list, is_query=False)
        self.triple_examplars_embeddings = torch.tensor(self.triple_examplars_embeddings)

        chain_examplars_text_list = [item["question"] for item in self.final_chain_examplars]
        self.chain_examplars_embeddings = self.encoder.encode(chain_examplars_text_list, is_query=True)
        self.chain_examplars_embeddings = torch.tensor(self.chain_examplars_embeddings)

        if self.triple_load_path is not None:
            with open(self.triple_load_path, "r") as f:
                self.extracted_doc_triples = json.load(f)
        else:
            self.extracted_doc_triples = {}  # id: triples
        self.token_id_to_choice_map = None

    def get_examplars_for_triple(self, doc_list, batch_size=64):
        # load demonstrations for each doc
        doc_examplars = []

        doc_text_list = []
        for doc_content in doc_list:
            title = doc_content.split("\n")[0]
            text = "\n".join(doc_content.split("\n")[1:])
            doc_text_list.append(f"title: {title} text: {text}")

        doc_embeddings = []
        for idx in range(0, len(doc_text_list), batch_size):
            batch_data = doc_text_list[idx : idx + batch_size]
            batch_embedding = self.encoder.encode(batch_data, is_query=True)
            doc_embeddings.append(batch_embedding)
        doc_embeddings = np.concatenate(doc_embeddings, axis=0)
        doc_embeddings = torch.tensor(doc_embeddings)

        similarities = torch.matmul(doc_embeddings, self.triple_examplars_embeddings.T)
        examplars_rank = torch.argsort(similarities, dim=1, descending=True)
        for i, _ in enumerate(doc_list):
            rank = examplars_rank[i].tolist()
            examplars = [self.triple_examplars[idx] for idx in rank[: self.num_examplars]]
            examplars = [
                "Title: {}\nText: {}\nKnowledge Triples: {}".format(
                    example["title"], example["text"], example["triples"]
                )
                for example in examplars
            ]
            doc_examplars.append(examplars)

        return doc_examplars

    def get_examplars_for_reasoning_chain(self, all_query):
        generating_chain_examplars = []
        final_chain_examplars = []
        query_embeddings = self.encoder.encode(all_query, is_query=True)
        query_embeddings = torch.tensor(query_embeddings)

        similarities = torch.matmul(query_embeddings, self.chain_examplars_embeddings.T)
        examplars_rank = torch.argsort(similarities, dim=1, descending=True)
        for i, _ in enumerate(all_query):
            rank = examplars_rank[i].tolist()
            examplars = [self.final_chain_examplars[idx] for idx in rank[: self.num_examplars]]
            final_chain_examplars.append(examplars)
            examplars = [self.generating_chain_examplars[idx] for idx in rank[: self.num_examplars]]
            generating_chain_examplars.append(examplars)
        return generating_chain_examplars, final_chain_examplars

    def parse_triple_output(self, doc_list, output_list):
        def parse_model_output(triples_text: str):
            import re

            results = []
            for one_triple_text in re.findall(r"<([^>]*)>", triples_text):
                pieces = one_triple_text.rsplit(";", maxsplit=2)
                if len(pieces) != 3:
                    print(f'Something wrong with this triple: "{one_triple_text}", Skip this triple!')
                    continue
                head, relation, tail = pieces
                results.append((head.strip(), relation.strip(), tail.strip()))
            return results

        # parse model_outputs
        results = []
        for j, (doc_content, generated_content) in enumerate(zip(doc_list, output_list)):
            triples = parse_model_output(generated_content)  # [(head, relation, tail)]
            triples_in_one_document = []
            title = doc_content.split("\n")[0]
            text = "\n".join(doc_content.split("\n")[1:])
            for head, relation, tail in triples:
                # if head.lower() != title.lower():
                #     if head.lower() not in text.lower():
                #         head = title

                triples_in_one_document.append(
                    {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                    }
                )
            results.append(triples_in_one_document)
        return results

    def extract_document_triples(self, queries, retrieval_results):
        """
        Extract triples from documents associated with each query, handling duplicates and generating prompts for LLM processing.
        """

        # deduplicate documents and map document IDs to their contents and associated queries
        unique_docs = {}
        doc_queries = {}
        for query, docs in zip(queries, retrieval_results):
            for doc in docs:
                # if 'id' not in doc:
                #     doc['id'] = hash_object(doc['contents'])
                doc["id"] = hash_object(doc["contents"])
                doc_id = doc["id"]
                if doc_id not in self.extracted_doc_triples:
                    unique_docs[doc_id] = doc["contents"]
                    doc_queries.setdefault(doc_id, []).append(query)

        # prepare data structures for document processing
        doc_ids = list(unique_docs.keys())
        doc_id_mapping = {doc_id: index for index, doc_id in enumerate(doc_ids)}
        docs_content = [unique_docs[doc_id] for doc_id in doc_ids]

        if len(docs_content) > 0:
            # obtain exemplars for each document
            doc_examplars = self.get_examplars_for_triple(docs_content)

            # construct prompts for triple extraction using an LLM
            # prompts for extracting triples from documents
            system_prompt = (
                "Given a title and a text, extract all the knowledge triples in the form of <title; relation; tail entity>, "
                "where title is the provided title, tail entity is a phrase in the text and relation denotes a description of the relation "
                "between the title and the tail entity. \n\nThe followings are some examples: \n\n{examplars}"
            )
            user_prompt = "Title: {title}\nText: {text}\nKnowledge Triples: "
            prompt_template = PromptTemplate(config=self.config, system_prompt=system_prompt, user_prompt=user_prompt)
            prompts = [
                prompt_template.get_string_with_varying_examplars(
                    question="",
                    examplars=examplars,
                    title=doc.split("\n")[0],
                    text="\n".join(doc.split("\n")[1:]),
                    tokenizer=self.generator.tokenizer,
                    max_length=2048,
                )
                for doc, examplars in zip(docs_content, doc_examplars)
            ]

            # generate triples via the language model
            outputs = self.generator.generate(prompts, max_tokens=512)
            triples = self.parse_triple_output(docs_content, outputs)

        # reconstruct triples for the original query-documents structure
        all_doc_triples = []
        for query_retrieval_result in retrieval_results:
            query_triples = []
            for doc in query_retrieval_result:
                doc_id = doc["id"]
                if doc_id in doc_id_mapping:
                    triple_set = triples[doc_id_mapping[doc_id]]
                    self.extracted_doc_triples[doc_id] = triple_set
                elif doc_id in self.extracted_doc_triples:
                    triple_set = self.extracted_doc_triples[doc_id]
                else:
                    raise AssertionError("Document ID not found during triple extraction.")
                query_triples.append(triple_set)
            all_doc_triples.append(query_triples)

        return all_doc_triples

    def convert_candidate_triples_to_choices(self, candidates):
        return "\n".join(
            ["A. no need for additional knowledge triples"]
            + ["{}. {}".format(chr(ord("B") + k), triple) for k, triple in enumerate(candidates)]
        )

    def build_prompt_for_reasoning_chain(
        self,
        hop,
        question,
        existing_triples,
        candidate_triples,
        generating_chain_examplars=[],
        final_chain_examplars=[],
        use_demonstration=True,
    ):

        base_instruction = (
            "Select the next knowledge triple that extends an existing set of knowledge triples to form a coherent reasoning path capable of answering a specified question. "
            "If the current reasoning path is sufficient to answer the question, simply output A. Please only output the choice for the next knowledge triple."
        )

        if use_demonstration and len(generating_chain_examplars) > 0:
            demonstration_instruction = (
                "\n\nThe followings are some examples of coherent reasoning paths capable of answering the specified question "
                f"and how the {hop}-th knowledge triples in these paths are selected:\n\n"
            )

            # deal with examplars
            examplars = []
            for i, (rp_examplar, grp_examplar) in enumerate(zip(final_chain_examplars, generating_chain_examplars)):
                if len(grp_examplar) < hop + 1:
                    continue
                examplar = "coherent reasoning path: {}\nquestion: {}\n".format(
                    rp_examplar["chains"], rp_examplar["question"]
                )
                examplar += "The {}-th triple in the reasoning path is selected as:\n".format(hop + 1)
                one_step_item = grp_examplar[hop]
                examplar += "existing knowledge triples: {}\nquestion: {}\ncandidate knowledge triples:\n{}\nthe next possible triple is:{}\n".format(
                    ", ".join(one_step_item["triples"]),
                    one_step_item["question"],
                    "\n".join(one_step_item["candidate_triples"]),
                    one_step_item["answer"],
                )
                examplars.append(examplar)
                if len(examplars) >= self.num_examplars:
                    break

            system_prompt = base_instruction + " " + demonstration_instruction + "{examplars}"
        else:
            system_prompt = base_instruction + "\n\n"

        user_prompt = (
            "The {hop}-th triple in the reasoning path is selected as:\nexisting knowledge triples: {existing_triples}\n"
            "question: {question}\ncandidate knowledge triples:\n{candidate_triples}\nthe next possible triple is:"
        )

        prompt_template = PromptTemplate(config=self.config, system_prompt=system_prompt, user_prompt=user_prompt)
        prompt = prompt_template.get_string_with_varying_examplars(
            hop=hop + 1,
            question=question,
            examplars=examplars,
            existing_triples=", ".join(existing_triples),
            candidate_triples=self.convert_candidate_triples_to_choices(candidate_triples),
            tokenizer=self.generator.tokenizer,
            max_length=2048,
        )

        return prompt

    def get_answer_token_indices(self, tokenizer, num_choices, token_ids):
        """Obtain the index of token corresponsding to the option"""
        if self.token_id_to_choice_map is None:
            self.token_id_to_choice_map = {}
            choices = [chr(ord("A") + i) for i in range(num_choices + 1)]
            for choice in choices:
                self.token_id_to_choice_map[tokenizer.encode(choice, add_special_tokens=False)[0]] = choice
                self.token_id_to_choice_map[tokenizer.encode(" {}".format(choice), add_special_tokens=False)[-1]] = (
                    choice
                )

        answer_token_indices = torch.zeros((token_ids.shape[0],), dtype=token_ids.dtype).fill_(token_ids.shape[1] - 1)
        for i in range(token_ids.shape[0]):
            for j in range(token_ids.shape[1]):
                if token_ids[i, j].item() in self.token_id_to_choice_map:
                    answer_token_indices[i] = j
                    break

        return answer_token_indices

    def get_reasoning_chain(self, all_query, all_doc_triples, triple_to_doc_ids):
        all_generating_chain_examplars, all_final_chain_examplars = self.get_examplars_for_reasoning_chain(all_query)
        all_chain_results = []

        for query, doc_triples, generating_chain_examplars, final_chain_examplars, triple_doc_id_list in tqdm(
            zip(
                all_query, all_doc_triples, all_generating_chain_examplars, all_final_chain_examplars, triple_to_doc_ids
            ),
            total=len(all_query),
            desc="Generating reasoning chain for query",
        ):
            # run single item
            flatten_triples = sum(doc_triples, [])  # all triples
            flatten_doc_ids = sum(
                [
                    [doc_id for _ in single_doc_triple]
                    for doc_id, single_doc_triple in zip(triple_doc_id_list, doc_triples)
                ],
                [],
            )
            num_total_triples = len(flatten_triples)
            triple_text = [
                "<{}; {}; {}>".format(triple_item["head"], triple_item["relation"], triple_item["tail"])
                for triple_item in flatten_triples
            ]
            triple_embeddings = torch.tensor(self.encoder.encode(triple_text, is_query=False))

            paths = [[]]  # each list contains index of triples to format a reasoning path
            paths_scores = [1.0]
            paths_finished = [False]

            for j in range(self.max_chain_length):
                if np.sum(paths_finished) == self.num_chains:
                    break

                # triple rank: concatenation of question and selected triples as query
                path_queries = [
                    "knowledge triples: {}\nquestion: {}".format(" ".join([triple_text[idx] for idx in path]), query)
                    for path in paths
                ]
                path_query_embeddings = torch.tensor(self.encoder.encode(path_queries, is_query=True))

                path_triples_similarities = torch.matmul(path_query_embeddings, triple_embeddings.T)
                candidate_triples_mask = torch.ones_like(path_triples_similarities)
                for k, path in enumerate(paths):
                    # mask the chosen triples
                    candidate_triples_mask[k, path] = 0.0
                path_triples_similarities = path_triples_similarities - 10000 * (1.0 - candidate_triples_mask)
                topk_most_relevant_triples_indices = torch.topk(
                    path_triples_similarities, k=min(self.topk_triple_select, num_total_triples), dim=1
                )[1].tolist()

                # construct prompts for selecting triple in reasoning chain
                # for each path, create an input to LLM
                input_prompts = []
                exisiting_triples = [
                    [triple_text[idx] for idx in path] for path in paths
                ]  # each item represents existing triples path
                candidate_triples = [
                    [triple_text[idx] for idx in candidate_triples_indices]
                    for candidate_triples_indices in topk_most_relevant_triples_indices
                ]  # each item represents candidate triples in a path
                for triples, candidates in zip(exisiting_triples, candidate_triples):
                    prompt = self.build_prompt_for_reasoning_chain(
                        hop=j,
                        question=query,
                        existing_triples=triples,
                        candidate_triples=candidates,
                        generating_chain_examplars=generating_chain_examplars,
                        final_chain_examplars=final_chain_examplars,
                        use_demonstration=True,
                    )
                    input_prompts.append(prompt)

                # get generated result
                torch.cuda.empty_cache()
                generate_output = self.generator.generate(input_prompts, max_tokens=32, return_dict=True)
                generated_token_ids, generated_token_logits = (
                    generate_output["generated_token_ids"],
                    generate_output["generated_token_logits"],
                )

                answer_token_indices = self.get_answer_token_indices(
                    self.generator.tokenizer, self.num_choices, generated_token_ids
                )
                answer_token_logits = generated_token_logits.gather(
                    1, answer_token_indices[:, None, None].expand(-1, -1, generated_token_logits.shape[-1])
                )
                answer_token_logits = answer_token_logits.squeeze(1)

                choices_token_ids_list = list(self.token_id_to_choice_map.keys())
                choices_list = [
                    self.token_id_to_choice_map[token_id] for token_id in choices_token_ids_list
                ]  # ['A','B','C',..], may be duplication
                answer_token_probs = F.softmax(answer_token_logits[:, choices_token_ids_list], dim=1)

                new_paths, new_paths_scores, new_paths_finished = [], [], []
                topk_choices_probs, topk_choices_indices = torch.topk(
                    answer_token_probs, k=self.num_beams, dim=1
                )  # for each path, select choice token in topk probs
                for i in range(len(paths)):
                    if paths_finished[i]:
                        new_paths.append(paths[i])
                        new_paths_scores.append(paths_scores[i])
                        new_paths_finished.append(True)
                        continue
                    if torch.all(torch.isnan(topk_choices_probs[i])):
                        print(
                            "No choice in generated results! generated text: {}".format(
                                self.generator.tokenizer.decode(generated_token_ids[i])
                            )
                        )
                        new_paths.append(paths[i])
                        new_paths_scores.append(paths_scores[i])
                        new_paths_finished.append(False)
                        continue
                    for b in range(self.num_beams):
                        if (
                            torch.isnan(topk_choices_probs[i, b])
                            or topk_choices_probs[i, b].item() < self.min_triple_prob
                        ):
                            continue
                        current_choice = choices_list[topk_choices_indices[i, b].item()]
                        if current_choice != "A" and (
                            ord(current_choice) - ord("B") >= len(topk_most_relevant_triples_indices[i])
                        ):
                            # generated invalid option
                            continue
                        new_paths_scores.append(paths_scores[i] * topk_choices_probs[i, b].item())
                        if current_choice == "A":
                            new_paths.append(paths[i] + [-1])
                            new_paths_finished.append(True)
                        else:
                            new_paths.append(
                                paths[i] + [topk_most_relevant_triples_indices[i][ord(current_choice) - ord("B")]]
                            )
                            new_paths_finished.append(False)

                assert len(new_paths) == len(new_paths_scores)
                assert len(new_paths) == len(new_paths_finished)
                new_paths_sorted_indices = sorted(
                    range(len(new_paths_scores)), key=lambda x: new_paths_scores[x], reverse=True
                )
                topk_new_paths_sorted_indices = new_paths_sorted_indices[: self.num_chains]
                paths = [new_paths[idx] for idx in topk_new_paths_sorted_indices]
                paths_scores = [new_paths_scores[idx] for idx in topk_new_paths_sorted_indices]
                paths_finished = [new_paths_finished[idx] for idx in topk_new_paths_sorted_indices]

            query_chain_result = [
                {
                    "triples": [triple_text[idx] for idx in path if idx >= 0],
                    "triple_doc_ids": [flatten_doc_ids[idx] for idx in path if idx >= 0],
                    "score": path_score,
                }
                for path, path_score in zip(paths, paths_scores)
            ]  # Contains paths with their scores

            # sort with scores
            query_chain_result.sort(key=lambda x: float(x["score"]), reverse=True)

            all_chain_results.append(query_chain_result)

        return all_chain_results

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            if self.reference_template is not None:
                format_reference += self.reference_template.format(idx=idx, title=title, text=text)
            else:
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

    def batch_run(self, dataset):
        all_query = dataset.question
        retrieval_results = dataset.retrieval_result
        print("Begin extracting triples")
        all_doc_triples = self.extract_document_triples(all_query, retrieval_results)
        triple_to_doc_ids = [
            [doc_item["id"] for doc_item in query_retrieval_result] for query_retrieval_result in retrieval_results
        ]
        dataset.update_output("doc_triples", all_doc_triples)
        # save triples for re-use
        with open(self.triple_save_path, "w") as f:
            json.dump(self.extracted_doc_triples, f, indent=4)
        print("Finish extracting tiples")

        print("Begin generating reasoning chain")
        reasoning_chain_result = self.get_reasoning_chain(all_query, all_doc_triples, triple_to_doc_ids)
        dataset.update_output("reasoning_chain", reasoning_chain_result)
        print("Finish generating reasoning chain")

        # get refine result based on context type
        refine_result = []

        if self.context_type == "triples":
            for query_chain_result in reasoning_chain_result:
                # query_chain_list: reasoning chain for a query
                query_chain_result = query_chain_result[: self.n_context]
                all_triple_text = []
                for chain in query_chain_result:
                    triples = chain["triples"]
                    for triple in triples:
                        triple = triple.replace("<", "").replace(">", "").replace(";", "", 2)
                        if triple not in all_triple_text:
                            all_triple_text.append(triple)
                refine_text = "\n".join(["{}. {}".format(i + 1, text) for i, text in enumerate(all_triple_text)])
                refine_result.append(refine_text)

        elif self.context_type == "triple-doc":
            for query_chain_result, query_retrieval_results in zip(reasoning_chain_result, retrieval_results):

                query_chain_result = query_chain_result[: self.n_context]
                chains_doc_id_count_dict = {}

                for chain in query_chain_result:
                    for triple, doc_id in zip(chain["triples"], chain["triple_doc_ids"]):
                        chains_doc_id_count_dict[doc_id] = chains_doc_id_count_dict.get(doc_id, 0) + 1

                ranked_chains_doc_id = sorted(chains_doc_id_count_dict.items(), key=lambda x: x[1], reverse=True)
                query_doc_to_idx = {doc_item["id"]: idx for idx, doc_item in enumerate(query_retrieval_results)}
                final_doc_idx = [query_doc_to_idx[doc_id] for doc_id in ranked_chains_doc_id]
                final_doc_list = [query_retrieval_results[idx] for idx in final_doc_idx]
                refine_text = self.format_reference(final_doc_list)
                refine_result.append(refine_text)

        else:
            assert False

        return refine_result
