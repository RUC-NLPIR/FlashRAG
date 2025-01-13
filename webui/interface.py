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
from components.index_builder import create_index_builder

def create_ui() -> "gr.Blocks":
    engine = Engine()

    with gr.Blocks(title = "⚡FlashRAG") as demo:
        gr.HTML("<h1><center>⚡FlashRAG: A Python Toolkit for Efficient RAG Research</center></h1>")
        
        gr.HTML(
            '<h3><center>Visit <a href="https://github.com/RUC-NLPIR/FlashRAG" target="_blank">'
            "FlashRAG</a> for details.</center></h3>"
        )

        engine.manager.add_elems("basic", create_basic(engine = engine))
        engine.manager.add_elems("retrieve", create_retrieve())
        engine.manager.add_elems("rerank", create_rerank())
        engine.manager.add_elems("generate", create_generator())
        engine.manager.add_elems("method", create_method())
        engine.manager.add_elems("preview", create_preview(engine = engine))
        engine.manager.add_elems("chat", create_chat(engine = engine))
        engine.manager.add_elems("evaluate", create_evaluate(engine = engine))
        
        lang: "gr.Dropdown" = engine.manager.get_elem_by_id("basic.lang")
        config_file: "gr.Dropdown" = engine.manager.get_elem_by_id("basic.config_file")
        config_save_btn: "gr.Button" = engine.manager.get_elem_by_id("preview.config_save_btn")
        output_box = engine.manager.get_elem_by_id("preview.output_box")
        
        demo.load(
            fn = engine.resume,
            outputs = engine.manager.get_elem_list(),
            concurrency_limit = None
        )
        
        lang.change(
            fn = engine.change_lang,
            inputs = [lang],
            outputs = engine.manager.get_elem_list(),
            queue = False
        )
        
        config_file.change(
            fn = engine.runner.load_config_from_file,
            inputs = [config_file],
            outputs = engine.manager.get_elem_list(),
            queue = True
        )
        
        config_save_btn.click(
            fn = engine.runner.save_configs,
            inputs = set(engine.manager.get_elem_list_without_accordion()),
            outputs = output_box,
            concurrency_limit = None
        ).then(
            engine.runner.update_config_file_list,
            [],
            config_file
        )
        
    return demo

if __name__ == "__main__":
    create_ui().launch()