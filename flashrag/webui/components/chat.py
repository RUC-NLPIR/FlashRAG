from gradio.components import Component
from typing import Dict
from components.constants import METHODS

import json
import gradio as gr

def create_chat() -> Dict[str, "Component"]:
    chatbot = gr.Chatbot(show_copy_button = True)
    messages = gr.State(value = [])
    
    with gr.Row():
        with gr.Column(scale = 5):
            query = gr.Textbox(
                show_label = False, interactive = True)
        with gr.Column(scale = 1):
            submit_btm = gr.Button(variant = "primary")
    
    return dict(
        chatbot = chatbot,
        messages = messages,
        query = query,
        submit_btn = submit_btm
    )
    