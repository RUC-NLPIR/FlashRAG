import re
from termcolor import colored

class AgentUtilsBase():
    def __init__(self):
        pass

    def truncate(self, tokenizer, tokenized_prompt, model_max_length):
        half = int(model_max_length/2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=False) + \
                tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=False)
        
        return prompt

    def preprocess_query(self, query):
        if "'" in query and '"' in query:
            query = query.replace("'", "\\'").replace('"', '\\"')
        return query

    def extract_code(self, text: str) -> str:
        triple_match = re.search(r'```[^\n]*\n(.+?)```', text, re.DOTALL)
        single_match = re.search(r'`([^`]*)`', text, re.DOTALL)
        if triple_match:
            return triple_match.group(1)
        elif single_match:
            return single_match.group(1)
        return text
    
    def postprocess_agent_response(self, response):
        """
        Implement the postprocess_agent_response function
        """
        raise NotImplementedError("postprocess_agent_response function must be implemented in a subclass")

class AgentUtils(AgentUtilsBase):
    def __init__(self):
        super().__init__()

    def parse_reasoning_steps(self, text: str):
        """
        Parse a string containing Thought/Action/Observation steps (including multi-line)
        and return a list of dictionaries of the form:
        
        [
            {
                "1": {
                    "Thought": "...",
                    "Action": "...",  # Only content inside backticks (if present)
                    "Observation": "..."
                }
            },
            {
                "2": {
                    "Thought": "...",
                    "Action": "...",
                    "Observation": "..."
                }
            },
            ...
        ]
        """
        # Regex pattern to match lines that start with "Thought X:", "Action X:", or "Observation X:".
        pattern = re.compile(r'^(Thought|Action|Observation)\s+(\d+):', re.MULTILINE)

        # This dictionary will accumulate:
        # data_dict[step_number] = {"Thought": ..., "Action": ..., "Observation": ...}
        data_dict = {}

        # We'll track the current label (Thought/Action/Observation) and step number
        current_label = None
        current_step = None
        last_pos = 0

        # Find all pattern occurrences in the text
        matches = list(pattern.finditer(text))

        for match in matches:
            # If we already have a label in progress, we can record its content
            if current_label is not None:
                # Slice the text from the last match's end to the start of this new match
                content = text[last_pos:match.start()].strip()
                # Store that content in data_dict
                data_dict[current_step][current_label] = content

            # Extract the new label and step
            label = match.group(1)       # "Thought", "Action", or "Observation"
            step = match.group(2)        # e.g. "1", "2", "3"

            # Ensure a dict for this step
            if step not in data_dict:
                data_dict[step] = {"Thought": None, "Action": None, "Observation": None}

            # Update current label/step
            current_label = label
            current_step = step
            # We'll slice from here next time
            last_pos = match.end()

        # Handle the final block after the last match
        if current_label is not None:
            content = text[last_pos:].strip()
            data_dict[current_step][current_label] = content

        # Post-process:
        #  - For each step, extract only the text inside triple backticks for "Action".
        for step_number in data_dict:
            action_text = data_dict[step_number]["Action"]
            if action_text:
                # Extract content inside triple backticks
                data_dict[step_number]["Action"] = self.extract_code(action_text)

        # Convert our dictionary to the desired list-of-dicts structure
        structured_data = []
        for step_number in sorted(data_dict.keys(), key=lambda x: int(x)):
            structured_data.append({step_number: data_dict[step_number]})

        return structured_data
    
    def postprocess_agent_response(self, response):
        """
        Extract Thought and Action, then extract the dict from Action.

        Return:
        - thought: str
        - action: dict
        - is_valid: bool
        """
        parsed_codes = self.parse_reasoning_steps(response)

        thoughts = []
        actions = []
        for steps in parsed_codes:
            for step_idx, step in steps.items():
                thought = f"Thought {step_idx}: {step['Thought']}"
                action = eval(self.extract_code(step['Action']))

                assert "function" in action, f"Action does not contain 'function' key: {action}"
                assert "parameters" in action, f"Action does not contain 'parameters' key: {action}"

                thoughts.append(thought)
                actions.append(action)

        return thoughts, actions
    
def print_code(codes):
    for idx, step in enumerate(codes):
        print(f"{colored(step['thought'], 'blue')}")
        print(colored(f"Action {idx+1}:\n```\n{step['action']}\n```", 'red'))
        print(colored(f"Observation {idx+1}: {step['observation']}", 'yellow'))
        # print(f"{'-'*60}")
