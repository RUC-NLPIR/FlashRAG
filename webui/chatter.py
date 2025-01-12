from manager import Manager
from runner import Runner

import gradio as gr
import time 

class Chatter:
    def __init__(
        self,
        runner: "Runner",
        manager: "Manager"
    ):
        self.runner = runner
        self.manager = manager
        
    def append(
        self,
        message,
        chatbot,
    ):
        if message["text"] is not None:
            return chatbot + [[message["text"], None]]
    
    def output(
        self,
        message,
        chatbot,
    ):
        base_output = ""
        for output in self.runner.pipeline.chat(query = message['text']):            
            base_output += "\n"

            for i in range(len(output)):
                time.sleep(0.001)
                chatbot[-1][1] = base_output + output[:i]
                yield chatbot, gr.update(value = None)
            base_output += output