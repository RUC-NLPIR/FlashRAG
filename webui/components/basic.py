from gradio.components import Component
from typing import Dict
from components.constants import METHODS
from engine import Engine
import gradio as gr

def create_basic(engine: "Engine") -> Dict[str, "Component"]:
    with gr.Row():
        lang = gr.Dropdown(
            choices = ["zh", "en"],
        )
        
        method_name = gr.Dropdown(
            choices = METHODS,
            value = METHODS[0],
            interactive = True
        )
        
        config_file = gr.Dropdown(
            choices = engine.runner.get_config_files(),
            value = None,
            interactive = True
        )
    
    with gr.Accordion(open = True) as base_tab:
        with gr.Row():
            gpu_id = gr.Textbox(interactive = True, placeholder = "0,1,2,3,4,5,6,7,8")
            framework = gr.Dropdown(
                choices = ["hf", "vllm", 'fschat', 'openai'],
                value = 'hf',
                interactive = True
            )

            generator_name = gr.Textbox(interactive = True, placeholder = 'llama3.1-8b-instruct')
            generator_model_path = gr.Textbox(interactive = True, placeholder = 'meta-llama/Llama-3.1-8B-Instruct')
        
        with gr.Row():
            retrieval_method = gr.Textbox(interactive = True, placeholder = 'e5')
            retrieval_model_path = gr.Textbox(interactive = True, placeholder = 'intfloat/e5-base-v2')

        with gr.Row():
            corpus_path = gr.Textbox(interactive = True)
            index_path = gr.Textbox(interactive = True)
        
    return dict(
        lang = lang,
        method_name = method_name,
        gpu_id = gpu_id,
        base_tab = base_tab,
        framework = framework,
        generator_name = generator_name,
        generator_model_path = generator_model_path,
        retrieval_method = retrieval_method,
        retrieval_model_path = retrieval_model_path,
        corpus_path = corpus_path,
        index_path = index_path,
        config_file = config_file
    )
    
    