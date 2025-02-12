from gradio.components import Component
from typing import Dict
from components.constants import METHODS

import json
import gradio as gr

def create_rerank() -> Dict[str, "Component"]:
    with gr.Accordion(open = False) as rerank_tab:
        with gr.Row():
            use_rerank = gr.Checkbox(interactive = True)
            rerank_model_name = gr.Textbox(interactive = True)
            rerank_model_path = gr.Textbox(interactive = True)
        
        with gr.Row():
            rerank_pooling_method = gr.Dropdown(choices = ["mean", "cls", 'pooling'], interactive = True)
            rerank_topk = gr.Slider(minimum = 1, maximum = 100, value = 10, step = 1, interactive = True)
            rerank_max_len = gr.Slider(minimum = 1, maximum = 2048, value = 512, step = 1, interactive = True)
            rerank_use_fp16 = gr.Checkbox(interactive = True)
        
    return dict(
        rerank_tab = rerank_tab,
        use_rerank = use_rerank,
        rerank_model_name = rerank_model_name,
        rerank_model_path = rerank_model_path,
        rerank_pooling_method = rerank_pooling_method,
        rerank_topk = rerank_topk,
        rerank_max_len = rerank_max_len,
        rerank_use_fp16 = rerank_use_fp16,
    )

    