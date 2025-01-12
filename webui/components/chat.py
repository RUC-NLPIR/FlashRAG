from gradio.components import Component
from typing import Dict
from engine import Engine

import gradio as gr

def create_chat(engine: "Engine") -> Dict[str, "Component"]:
    with gr.Tab() as chat_tab:
        chatbot = gr.Chatbot(show_copy_button = True)
        input_elems = set(engine.manager.get_elem_list_without_accordion())

        with gr.Row():
            query = gr.MultimodalTextbox(
                interactive = True,
                show_label = False
            )
        
    query.submit(
        engine.chatter.append,
        inputs = [query, chatbot],
        outputs = chatbot
    ).then(
        engine.runner.load_pipeline,
        inputs = input_elems,
        outputs = chatbot,
    ).then(
        engine.chatter.output,
        inputs = [query, chatbot],
        outputs = [chatbot, query]
    )
    
    return dict(
        chat_tab = chat_tab,
        chatbot = chatbot,
        query = query
    )