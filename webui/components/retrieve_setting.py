from gradio.components import Component
from typing import Dict
from components.constants import METHODS

import json
import gradio as gr


def create_retrieve() -> Dict[str, "Component"]:
    with gr.Accordion(open=False) as retrieve_tab:
        with gr.Row():
            instruction = gr.Textbox(interactive=True)
            retrieval_topk = gr.Slider(
                minimum=1, maximum=100, value=5, step=1, interactive=True
            )
            retrieval_use_fp16 = gr.Checkbox(value=True, interactive=True)
            retrieval_pooling_method = gr.Dropdown(
                choices=["mean", "pooling", "cls"], interactive=True
            )
        with gr.Row():
            query_max_length = gr.Slider(
                minimum=1, maximum=2048, value=512, step=1, interactive=True
            )
            retrieval_batch_size = gr.Slider(
                minimum=1, maximum=1024, value=256, step=1, interactive=True
            )
        with gr.Row():
            bm25_backend = gr.Dropdown(
                choices=["bm25s", "pyserini"], value="pyserini", interactive=True
            )
            use_sentence_transformers = gr.Checkbox(value=False, interactive=True)
        with gr.Row():
            save_retrieval_cache = gr.Checkbox(value=False, interactive=True)
            use_retrieval_cache = gr.Checkbox(value=False, interactive=True)
            retrieval_cache_path = gr.Textbox(interactive=True)

    return dict(
        retrieve_tab=retrieve_tab,
        instruction=instruction,
        retrieval_topk=retrieval_topk,
        retrieval_use_fp16=retrieval_use_fp16,
        retrieval_pooling_method=retrieval_pooling_method,
        query_max_length=query_max_length,
        retrieval_batch_size=retrieval_batch_size,
        bm25_backend=bm25_backend,
        use_sentence_transformers=use_sentence_transformers,
        save_retrieval_cache=save_retrieval_cache,
        use_retrieval_cache=use_retrieval_cache,
        retrieval_cache_path=retrieval_cache_path,
    )
