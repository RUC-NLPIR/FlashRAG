import os
import gradio as gr
from engine import Engine
from components.basic import create_basic
from components.retrieve_setting import create_retrieve
from components.rerank_setting import create_rerank
from components.generate_setting import create_generator
from components.preview import create_preview
from components.method_setting import create_method
from components.chat import create_chat
from components.evaluate import create_evaluate

def create_ui() -> "gr.Blocks":
    engine = Engine()

    with gr.Blocks(title="FlashRAG") as demo:
        gr.HTML("<h1><center>⚡FlashRAG: A Python Toolkit for Efficient RAG Research</center></h1>")
        gr.HTML(
            '<h3><center>Visit <a href="https://github.com/RUC-NLPIR/FlashRAG" target="_blank">'
            "FlashRAG</a> for details.</center></h3>"
        )
    
        engine.manager.add_elems("basic", create_basic())
        engine.manager.add_elems("retrieve", create_retrieve())
        engine.manager.add_elems("rerank", create_rerank())
        engine.manager.add_elems("generate", create_generator())
        engine.manager.add_elems("method", create_method())
        engine.manager.add_elems("preview", create_preview(engine = engine))
        
        with gr.Tab("Chat"):
            engine.manager.add_elems("chat", create_chat())
        with gr.Tab('Evaluate'):
            engine.manager.add_elems("evaluate", create_evaluate())
        
        
        lang = engine.manager.get_elem_by_id("basic.lang")
        
        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        
        
    return demo

if __name__ == "__main__":
    create_ui().launch()
    #create_ui().launch(server_port = 10001)