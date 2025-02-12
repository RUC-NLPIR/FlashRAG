from gradio.components import Component
from typing import Dict
from components.constants import METRICS
from engine import Engine
import gradio as gr

def create_index_builder(engine: "Engine") -> Dict[str, "Component"]:
    pass