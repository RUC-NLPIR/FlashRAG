from gradio.components import Component
from typing import Dict
from components.constants import METHODS

import json
import gradio as gr

def create_generator() -> Dict[str, "Component"]:
    with gr.Accordion(open = False) as generate_tab:
        with gr.Accordion(open = False) as openai_tab:
            with gr.Row():
                api_key = gr.Textbox(
                    value = "",
                    interactive = True,
                    placeholder='sk-xxx'
                )
                base_url = gr.Textbox(
                    value = "",
                    interactive =True,
                    placeholder='https://api.openai.com/v1/chat'
                )
        
        with gr.Row():
            generator_max_input_len = gr.Textbox(
                value = "4096",
                interactive = True
            )
            generator_batch_size = gr.Number(
                value = 1,
                interactive = True
            )
            gpu_memory_utilization = gr.Slider(
                minimum = 0,
                maximum = 1,
                step = 0.01,
                value = 0.7,
                interactive = True
            )
            
        with gr.Row():
            generate_do_sample = gr.Checkbox(
                value = True,
                interactive = True
            )
            generate_max_new_tokens = gr.Number(
                value = 512,
                interactive = True
            )
            generate_use_fid = gr.Checkbox(
                value = False,
                interactive = True
            )
            
        with gr.Row():
            generate_temperature = gr.Slider(
                minimum = 0,
                maximum = 2,
                step = 0.01,
                value = 0.7,
                interactive = True
            )
            generate_top_p = gr.Slider(
                minimum = 0,
                maximum = 1,
                step = 0.01,
                value = 0.9,
                interactive = True
            )
            generate_top_k = gr.Number(
                value = -1,
                interactive = True
            )
            
    return dict(
        generate_tab = generate_tab,
        openai_tab = openai_tab,
        api_key = api_key,
        base_url = base_url,
        generator_max_input_len = generator_max_input_len,
        generator_batch_size = generator_batch_size, 
        gpu_memory_utilization = gpu_memory_utilization,
        generate_do_sample = generate_do_sample,
        generate_max_new_tokens = generate_max_new_tokens,
        generate_use_fid = generate_use_fid,
        generate_temperature = generate_temperature,
        generate_top_p = generate_top_p,
        generate_top_k = generate_top_k,
    )