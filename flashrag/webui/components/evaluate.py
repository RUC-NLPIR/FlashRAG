from gradio.components import Component
from typing import Dict
from components.constants import METRICS

import json
import gradio as gr


def create_evaluate() -> Dict[str, "Component"]:
    with gr.Row():
        data_dir = gr.Textbox(interactive = True)
        dataset_name = gr.Textbox(interactive = True)
        save_dir = gr.Textbox(interactive = True)
    with gr.Row():
        save_intermediate_data = gr.Checkbox(value=True, interactive = True)
        save_note = gr.Textbox(interactive = True)
        seed = gr.Textbox(interactive = True)
    with gr.Row():
        # TODO: auto download from huggingface
        # dataset_name = gr.Dropdown(
        #     choices = SUPPORTED_DATASETS,
        #     value = SUPPORTED_DATASETS[0],
        #     interactive = True
        # )
       
        test_sample_num = gr.Number(value = 10, interactive = True)
        random_sample = gr.Checkbox(value = False, interactive = True)    
    with gr.Row():
        use_metrics = gr.Dropdown(multiselect = True, choices = METRICS, value=METRICS[:4], interactive = True)
        save_metric_score = gr.Checkbox(value = True, interactive = True)

    return dict(
        data_dir = data_dir,
        save_dir = save_dir,
        save_intermediate_data = save_intermediate_data,
        save_note = save_note,
        seed = seed,
        dataset_name = dataset_name,
        test_sample_num = test_sample_num,
        random_sample = random_sample,
        save_metric_score = save_metric_score,
        use_metrics= use_metrics
    )