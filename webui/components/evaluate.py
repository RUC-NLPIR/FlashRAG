from gradio.components import Component
from typing import Dict
from components.constants import METRICS
from engine import Engine
import gradio as gr


def create_evaluate(engine: "Engine") -> Dict[str, "Component"]:
    input_pipeline_elems = set(engine.manager.get_elem_list_without_accordion())
    with gr.Tab() as evaluate_tab:
        with gr.Row():
            with gr.Column(scale = 3):
                with gr.Row():
                    data_dir = gr.Textbox(interactive = True)
                    dataset_name = gr.Dropdown(
                        choices = [],
                        value = None,
                        interactive = True
                    )
                    dataset_split = gr.Dropdown(
                        choices = [],
                        value = None,
                        interactive = True
                    )
                with gr.Row():
                    save_intermediate_data = gr.Checkbox(
                        value = True,
                        interactive = True
                    )
                    save_dir = gr.Textbox(interactive = True)
                    save_note = gr.Textbox(interactive = True)
                
                with gr.Row():
                    # TODO: auto download from huggingface
                    seed = gr.Textbox(interactive = True)
                    test_sample_num = gr.Number(
                        value = 10,
                        interactive = True
                    )
                    random_sample = gr.Checkbox(
                        value = False,
                        interactive = True
                    )   
                
                with gr.Row():
                    use_metrics = gr.Dropdown(
                        multiselect = True,
                        choices = METRICS,
                        value = METRICS[:4],
                        interactive = True
                    )
                    save_metric_score = gr.Checkbox(
                        value = True,
                        interactive = True
                    )
            
            with gr.Column(scale = 1):
                with gr.Row(variant = 'panel'):
                    evaluate_output_box = gr.Markdown(
                        height = 350,
                        show_copy_button = True
                    )
                with gr.Row():
                    evaluate_preview_btn = gr.Button(
                        value = None,
                        variant = 'primary',
                        size = 'lg',
                        interactive = True
                    )
                with gr.Row():
                    evaluate_run_btn = gr.Button(
                        value = None,
                        variant = 'huggingface',
                        size = 'lg',
                        interactive = True
                    )
        with gr.Row():
            terminal_info = gr.HTML()
            
        with gr.Row(variant = 'panel'):
            terminal = gr.Markdown(
                height = 500,
                show_copy_button = False,
                show_label = True,
                container = True
            )

    
    input_eval_elems = set([
        data_dir, dataset_name, dataset_split, save_dir, save_intermediate_data,
        save_note, seed, test_sample_num, random_sample, save_metric_score, use_metrics
    ])
    
    preivew_eval_elems = [evaluate_output_box]
    
    data_dir.change(
        fn = engine.runner.get_data_subfolders,
        inputs = data_dir,
        outputs = dataset_name
    )
    
    dataset_name.change(
        fn = engine.runner.get_dataset_split,
        inputs = [data_dir, dataset_name],
        outputs = dataset_split
    )
    
    evaluate_preview_btn.click(
        fn = engine.runner.preview_eval_configs,
        inputs = input_eval_elems,
        outputs = preivew_eval_elems,
    )
    
    evaluate_run_btn.click(
        fn = engine.runner.run_evaluate,
        inputs = input_pipeline_elems | input_eval_elems,
        outputs = terminal,
        show_progress = True,
    )

    return dict(
        evaluate_tab = evaluate_tab,
        data_dir = data_dir,
        dataset_split = dataset_split,
        save_dir = save_dir,
        save_intermediate_data = save_intermediate_data,
        save_note = save_note,
        seed = seed,
        terminal = terminal,
        terminal_info = terminal_info,
        dataset_name = dataset_name,
        test_sample_num = test_sample_num,
        random_sample = random_sample,
        save_metric_score = save_metric_score,
        use_metrics = use_metrics,
        evaluate_output_box = evaluate_output_box,
        evaluate_preview_btn = evaluate_preview_btn,
        evaluate_run_btn = evaluate_run_btn
    )