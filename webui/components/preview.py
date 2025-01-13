from gradio.components import Component
from typing import Dict, Set
from components.constants import METHODS
from engine import Engine

import json
import gradio as gr

def create_preview(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = set(engine.manager.get_elem_list_without_accordion())
    
    with gr.Row():
        config_preview_btn = gr.Button() 
        config_save_btn = gr.Button()

    with gr.Row(variant = "panel"):
        output_box = gr.Markdown(height = 300, show_copy_button = True)
    
    preview_output_elems = [output_box]
    
    config_preview_btn.click(
        engine.runner.preview_pipeline_configs,
        input_elems,
        preview_output_elems,
        concurrency_limit = None
    )

    return dict(
        config_preview_btn = config_preview_btn,
        config_save_btn = config_save_btn,
        output_box = output_box
    )