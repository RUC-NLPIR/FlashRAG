from typing import Dict, Any
import json
def format_text(text, underline = False, bold = False):
    if underline:
        text = f"<u>{text}</u>"
    if bold:
        text = f"<b>{text}</b>"
    return text
def gen_config(args: Dict[str, Any]) -> str:
    
    config_lines = json.dumps(args, indent = 4)
    # for i, (k, v) in enumerate(args.items()):
    #     config_lines += "{}: {}\t".format(k, v)
    #     if (i + 1) % 1 == 0:
    #         config_lines += "\n"
        
    return f"```bash\n{config_lines}\n```"