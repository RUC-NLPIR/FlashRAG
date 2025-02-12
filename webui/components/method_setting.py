from gradio.components import Component
from typing import Dict
from components.constants import METHODS

import json
import gradio as gr


def create_method() -> Dict[str, "Component"]:
    with gr.Tabs():
        with gr.TabItem("LLMLingua"):

            with gr.Row():
                llmlingua_info = gr.HTML()

            with gr.Accordion(open=False) as llmlingua_advanced:
                with gr.Row():
                    llmlingua_refiner_path = gr.Textbox(interactive=True)
                    llmlingua_use_llmlingua2 = gr.Checkbox(
                        value=False, interactive=True
                    )
                    llmlingua_refiner_input_prompt_flag = gr.Checkbox(
                        value=False, interactive=True
                    )
                with gr.Row():
                    llmlingua_rate = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.55, interactive=True
                    )
                    llmlingua_target_token = gr.Textbox(value=-1, interactive=True)
                    llmlingua_condition_in_question = gr.Textbox(
                        value="after_condition", interactive=True
                    )
                    llmlingua_reorder_context = gr.Textbox(
                        value="sort", interactive=True
                    )
                with gr.Row():
                    llmlingua_condition_compare = gr.Checkbox(
                        value=True, interactive=True
                    )
                    llmlingua_context_budget = gr.Textbox(
                        value="+100", interactive=True
                    )
                    llmlingua_rank_method = gr.Textbox(
                        value="longllmlingua", interactive=True
                    )
                with gr.Row():
                    llmlingua_force_tokens = gr.Textbox(
                        value="", interactive=True, placeholder="None"
                    )
                    llmlingua_chunk_end_tokens = gr.Textbox(
                        value="", interactive=True, placeholder="None"
                    )

        with gr.TabItem("Recomp"):
            with gr.Row():
                recomp_info = gr.HTML()

            with gr.Accordion(open=False) as recomp_advanced:
                with gr.Row():
                    recomp_refiner_path = gr.Textbox(interactive=True)
                    # for abstractive
                    recomp_max_input_length = gr.Slider(
                        minimum=1, maximum=4096, step=1, value=1024, interactive=True
                    )
                    recomp_max_output_length = gr.Slider(
                        minimum=1, maximum=2048, step=1, value=512, interactive=True
                    )
                with gr.Row():
                    # for extractive
                    recomp_topk = gr.Slider(
                        minimum=1, maximum=500, step=1, value=5, interactive=True
                    )
                    recomp_encode_max_length = gr.Slider(
                        minimum=1, maximum=2048, step=1, value=256, interactive=True
                    )
                    recomp_refiner_pooling_method = gr.Dropdown(
                        choices=["mean", "pooling", "cls"],
                        value="mean",
                        interactive=True,
                    )

        with gr.TabItem("Selective-Context"):
            with gr.Row():
                sc_info = gr.HTML()
            with gr.Accordion(open=False) as sc_advanced:
                with gr.Row():
                    sc_refiner_path = gr.Textbox(interactive=True)
                    sc_reduce_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, interactive=True
                    )
                    sc_reduce_level = gr.Dropdown(
                        choices=["sent", "phrase", "token"],
                        value="token",
                        interactive=True,
                    )

        with gr.TabItem("Ret-Robust"):
            with gr.Row():
                retrobust_info = gr.HTML()

            with gr.Accordion(open=False) as retrobust_advanced:
                with gr.Row():
                    retrobust_generator_lora_path = gr.Textbox(interactive=True)
                    retrobust_max_iter = gr.Slider(
                        minimum=1, maximum=7, step=1, value=5, interactive=True
                    )
                    retrobust_single_hop = gr.Checkbox(value=False, interactive=True)

        with gr.TabItem("SKR"):
            with gr.Row():
                skr_info = gr.HTML()

            with gr.Accordion(open=False) as skr_advanced:
                with gr.Row():
                    skr_judger_path = gr.Textbox(interactive=True)
                    skr_training_data_path = gr.Textbox(interactive=True)
                with gr.Row():
                    skr_topk = gr.Slider(
                        minimum=1, maximum=10, step=1, value=5, interactive=True
                    )
                    skr_batch_size = gr.Slider(
                        minimum=1, maximum=2048, step=1, value=64, interactive=True
                    )
                    skr_max_length = gr.Slider(
                        minimum=1, maximum=2048, step=1, value=128, interactive=True
                    )

        with gr.TabItem("Self-RAG"):
            with gr.Row():
                selfrag_info = gr.HTML()

            with gr.Accordion(open=False) as selfrag_advanced:
                with gr.Row():
                    selfrag_mode = gr.Dropdown(
                        choices=[
                            "adaptive_retrieval",
                            "always_retrieve",
                            "no_retrieval",
                        ],
                        value="adaptive_retrieval",
                        interactive=True,
                    )
                    selfrag_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.2, interactive=True
                    )
                    selfrag_max_depth = gr.Slider(
                        minimum=1, maximum=10, step=1, value=2, interactive=True
                    )
                    selfrag_beam_width = gr.Slider(
                        minimum=1, maximum=5, step=1, value=2, interactive=True
                    )
                with gr.Row():
                    selfrag_w_rel = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, interactive=True
                    )
                    selfrag_w_sup = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, interactive=True
                    )
                    selfrag_w_use = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, interactive=True
                    )
                with gr.Row():
                    selfrag_use_grounding = gr.Checkbox(value=True, interactive=True)
                    selfrag_use_utility = gr.Checkbox(value=True, interactive=True)
                    selfrag_use_seqscore = gr.Checkbox(value=True, interactive=True)
                    selfrag_ignore_cont = gr.Checkbox(value=False, interactive=True)

        with gr.TabItem("FLARE"):
            with gr.Row():
                flare_info = gr.HTML()

            with gr.Accordion(open=False) as flare_advanced:
                with gr.Row():
                    flare_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.2, interactive=True
                    )
                    flare_look_ahead_steps = gr.Slider(
                        minimum=1, maximum=256, step=1, value=64, interactive=True
                    )
                    flare_max_generation_length = gr.Slider(
                        minimum=1, maximum=2048, step=1, value=256, interactive=True
                    )
                    flare_max_iter_num = gr.Slider(
                        minimum=1, maximum=10, step=1, value=5, interactive=True
                    )

        with gr.TabItem("IterRetGen"):
            with gr.Row():
                iterretgen_info = gr.HTML()

            with gr.Accordion(open=False) as iterretgen_advanced:
                with gr.Row():
                    iterretgen_iter_num = gr.Slider(
                        minimum=1, maximum=10, step=1, value=3, interactive=True
                    )

        with gr.TabItem("IRCOT"):
            with gr.Row():
                ircot_info = gr.HTML()

            with gr.Accordion(open=False) as ircot_advanced:
                with gr.Row():
                    ircot_max_iter = gr.Slider(
                        minimum=1, maximum=7, step=1, value=5, interactive=True
                    )

        with gr.TabItem("Trace"):
            with gr.Row():
                trace_info = gr.HTML()

            with gr.Accordion(open=False) as trace_advanced:
                with gr.Row():
                    trace_num_examplars = gr.Slider(
                        minimum=1, maximum=10, step=1, value=3, interactive=True
                    )
                    trace_max_chain_length = gr.Slider(
                        minimum=1, maximum=10, step=1, value=4, interactive=True
                    )
                    trace_topk_triple_select = gr.Slider(
                        minimum=1, maximum=10, step=1, value=5, interactive=True
                    )
                    trace_num_choices = gr.Slider(
                        minimum=1, maximum=50, step=1, value=20, interactive=True
                    )
                with gr.Row():
                    trace_min_triple_prob = gr.Slider(
                        minimum=1e-4,
                        maximum=1e-3,
                        step=1e-4,
                        value=1e-4,
                        interactive=True,
                    )
                    trace_num_beams = gr.Slider(
                        minimum=1, maximum=20, step=1, value=5, interactive=True
                    )
                    trace_num_chains = gr.Slider(
                        minimum=1, maximum=50, step=1, value=20, interactive=True
                    )
                    trace_n_context = gr.Slider(
                        minimum=1, maximum=10, step=1, value=5, interactive=True
                    )
                    trace_context_type = gr.Dropdown(
                        choices=["triples", "triple-doc"],
                        value="triples",
                        interactive=True,
                    )

        with gr.TabItem("Spring"):
            with gr.Row():
                spring_info = gr.HTML()

            with gr.Accordion(open=False) as spring_advanced:
                with gr.Row():
                    spring_token_embedding_path = gr.Textbox(
                        interactive=True,
                        placeholder="llama2.13b.base.added_token_embeddings.pt",
                    )

        with gr.TabItem("Adaptive-RAG"):
            with gr.Row():
                adaptive_info = gr.HTML()

            with gr.Accordion(open=False) as adaptive_advanced:
                with gr.Row():
                    adaptive_judger_path = gr.Textbox(interactive=True)

        with gr.TabItem("RQ-RAG"):
            with gr.Row():
                rqrag_info = gr.HTML()

            with gr.Accordion(open=False) as rqrag_advanced:
                with gr.Row():
                    rqrag_max_depth = gr.Number(value=2, interactive=True)

    return dict(
        llmlingua_info=llmlingua_info,
        llmlingua_refiner_path=llmlingua_refiner_path,
        llmlingua_use_llmlingua2=llmlingua_use_llmlingua2,
        llmlingua_refiner_input_prompt_flag=llmlingua_refiner_input_prompt_flag,
        llmlingua_rate=llmlingua_rate,
        llmlingua_target_token=llmlingua_target_token,
        llmlingua_condition_in_question=llmlingua_condition_in_question,
        llmlingua_reorder_context=llmlingua_reorder_context,
        llmlingua_condition_compare=llmlingua_condition_compare,
        llmlingua_context_budget=llmlingua_context_budget,
        llmlingua_rank_method=llmlingua_rank_method,
        llmlingua_force_tokens=llmlingua_force_tokens,
        llmlingua_chunk_end_tokens=llmlingua_chunk_end_tokens,
        recomp_info=recomp_info,
        recomp_refiner_path=recomp_refiner_path,
        recomp_max_input_length=recomp_max_input_length,
        recomp_max_output_length=recomp_max_output_length,
        recomp_topk=recomp_topk,
        recomp_encode_max_length=recomp_encode_max_length,
        recomp_refiner_pooling_method=recomp_refiner_pooling_method,
        sc_info=sc_info,
        sc_refiner_path=sc_refiner_path,
        sc_reduce_ratio=sc_reduce_ratio,
        sc_reduce_level=sc_reduce_level,
        retrobust_info=retrobust_info,
        retrobust_generator_lora_path=retrobust_generator_lora_path,
        retrobust_max_iter=retrobust_max_iter,
        retrobust_single_hop=retrobust_single_hop,
        skr_info=skr_info,
        skr_judger_path=skr_judger_path,
        skr_training_data_path=skr_training_data_path,
        skr_topk=skr_topk,
        skr_batch_size=skr_batch_size,
        skr_max_length=skr_max_length,
        selfrag_info=selfrag_info,
        selfrag_mode=selfrag_mode,
        selfrag_threshold=selfrag_threshold,
        selfrag_max_depth=selfrag_max_depth,
        selfrag_beam_width=selfrag_beam_width,
        selfrag_w_rel=selfrag_w_rel,
        selfrag_w_sup=selfrag_w_sup,
        selfrag_w_use=selfrag_w_use,
        selfrag_use_grounding=selfrag_use_grounding,
        selfrag_use_utility=selfrag_use_utility,
        selfrag_use_seqscore=selfrag_use_seqscore,
        selfrag_ignore_cont=selfrag_ignore_cont,
        flare_info=flare_info,
        flare_threshold=flare_threshold,
        flare_look_ahead_steps=flare_look_ahead_steps,
        flare_max_generation_length=flare_max_generation_length,
        flare_max_iter_num=flare_max_iter_num,
        iterretgen_info=iterretgen_info,
        iterretgen_iter_num=iterretgen_iter_num,
        ircot_info=ircot_info,
        ircot_max_iter=ircot_max_iter,
        trace_info=trace_info,
        trace_num_examplars=trace_num_examplars,
        trace_max_chain_length=trace_max_chain_length,
        trace_topk_triple_select=trace_topk_triple_select,
        trace_num_choices=trace_num_choices,
        trace_min_triple_prob=trace_min_triple_prob,
        trace_num_beams=trace_num_beams,
        trace_num_chains=trace_num_chains,
        trace_n_context=trace_n_context,
        trace_context_type=trace_context_type,
        spring_info=spring_info,
        spring_token_embedding_path=spring_token_embedding_path,
        adaptive_info=adaptive_info,
        adaptive_judger_path=adaptive_judger_path,
        rqrag_info=rqrag_info,
        rqrag_max_depth=rqrag_max_depth,
        llmlingua_advanced=llmlingua_advanced,
        recomp_advanced=recomp_advanced,
        sc_advanced=sc_advanced,
        retrobust_advanced=retrobust_advanced,
        skr_advanced=skr_advanced,
        selfrag_advanced=selfrag_advanced,
        flare_advanced=flare_advanced,
        iterretgen_advanced=iterretgen_advanced,
        ircot_advanced=ircot_advanced,
        trace_advanced=trace_advanced,
        spring_advanced=spring_advanced,
        adaptive_advanced=adaptive_advanced,
        rqrag_advanced=rqrag_advanced,
    )
