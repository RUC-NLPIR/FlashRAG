from typing import Dict, Any
import json
import yaml

def format_text(
    text: str,
    underline: bool = False,
    bold: bool = False
) -> str:
    
    if underline:
        text = f"<u>{text}</u>"
    if bold:
        text = f"<b>{text}</b>"
        
    return text

def gen_config(args: Dict[str, Any]) -> str:
    config_lines = json.dumps(args, indent = 4)
    return f"```bash\n{config_lines}\n```"

def read_yaml_file(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return data
    
    except FileNotFoundError:
        print(f"File {file_path} does not exist.")
        return None
    
    except yaml.YAMLError as e:
        print(f"Parse YAML file error: {e}")
        return None

def flatten_dict(
    nested_dict: Dict[str, Any],
    parent_key: str = "",
    sep: str = "."
) -> str:
    
    flattened = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, sep=sep))
        else:
            flattened[new_key] = value
            
    return flattened

class TeeStream:
    def __init__(self, queue, original_stream):
        self.queue = queue
        self.original_stream = original_stream

    def write(self, message):
        self.original_stream.write(message)
        self.original_stream.flush()
        if message.strip():  
            self.queue.put(message)

    def flush(self):
        self.original_stream.flush()