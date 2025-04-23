from transformers import AutoTokenizer, AutoConfig
import tiktoken
import warnings

class PromptTemplate:
    placeholders = ["reference", "question"]
    base_system_prompt = (
        "Answer the question based on the given document."
        "Only give me the answer and do not output any other words."
        "\nThe following are given documents.\n\n{reference}"
    )
    base_user_prompt = "Question: {question}"

    def __init__(self, config, system_prompt="", user_prompt="", reference_template=None, enable_chat=True):

        self.config = config
        self.is_openai = config["framework"] == "openai"
        self.max_input_len = config['generator_max_input_len']
        if not self.is_openai:
            self.generator_path = config["generator_model_path"]
            model_config = AutoConfig.from_pretrained(self.generator_path, trust_remote_code=True)
            model_name = model_config._name_or_path.lower()
            self.is_chat = False
            if "chat" in model_name or "instruct" in model_name:
                self.is_chat = True
            self.tokenizer = AutoTokenizer.from_pretrained(self.generator_path, trust_remote_code=True)
        else:
            self.is_chat = True
            self.enable_chat = True
            try:
                self.tokenizer = tiktoken.encoding_for_model(config['generator_model'])
            except Exception as e:
                print("Error: ", e)
                warnings.warn("This model is not supported by tiktoken. Use gpt-3.5-turbo instead.")
                self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

        if len(system_prompt) == 0 and len(user_prompt) == 0:
            system_prompt = self.base_system_prompt
            user_prompt = self.base_user_prompt
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_chat = enable_chat
        self.reference_template = reference_template

        # self._check_placeholder()

    def _check_placeholder(self):
        # check placeholder in prompt
        for holder in self.placeholders:
            flag = False
            for prompt in [self.system_prompt, self.user_prompt]:
                if f"{holder}" in prompt:
                    print(f"Find `{holder}` in template")
                    flag = True
                    break
            if not flag and holder != "reference":
                assert False

    def truncate_prompt(self, prompt):
        if self.is_openai:
            if self.enable_chat:
                truncated_messages = []
                total_tokens = 0
                assert isinstance(prompt, list)
                for message in prompt:
                    role_content = message['content']
                    encoded_message = self.tokenizer.encode(role_content)

                    if total_tokens + len(encoded_message) <= self.max_input_len:
                        truncated_messages.append(message)
                        total_tokens += len(encoded_message)
                    else:
                        print(f"The input text length is greater than the maximum length ({total_tokens + len(encoded_message)} > {self.max_input_len}) and has been truncated!")
                        remaining_tokens = self.max_input_len - total_tokens
                        truncated_message = self.encoding.decode(encoded_message[:remaining_tokens])
                        message['content'] = truncated_message
                        truncated_messages.append(message)
                        break
            else:
                assert isinstance(prompt, str)
                tokenized_prompt = self.tokenizer.encode(prompt,allowed_special={'<|endoftext|>'})
                half = int(self.max_input_len / 2)
                truncated_messages = self.tokenizer.decode(tokenized_prompt[:half]) + self.tokenizer.decode(tokenized_prompt[-half:])

            return truncated_messages

        else:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.generator_path, trust_remote_code=True)
            assert isinstance(prompt, str)
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

            if len(tokenized_prompt) > self.max_input_len:
                print(f"The input text length is greater than the maximum length ({len(tokenized_prompt)} > {self.max_input_len}) and has been truncated!")
                half = int(self.max_input_len / 2)
                prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                        self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            return prompt



    def get_string(self, question=None, retrieval_result=None, formatted_reference=None, previous_gen=None, messages=None, **params):
        if messages is not None:
            if isinstance(messages, str):
                return self.truncate_prompt(messages)
            if self.is_chat and self.enable_chat:
                if self.is_openai:
                    self.truncate_prompt(messages)
                else:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    return self.truncate_prompt(prompt)
            else:
                prompt = "\n\n".join(
                    [message['content'] for message in messages if message['content']]
                )
                return self.truncate_prompt(prompt)

        if formatted_reference is None:
            if retrieval_result is not None:
                formatted_reference = self.format_reference(retrieval_result)
            else:
                formatted_reference = ""

        input_params = {"question": question, "reference": formatted_reference}
        input_params.update(**params)

        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        if self.is_chat and self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role": "user", "content": user_prompt})
            if not self.is_openai:
                input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])

        if previous_gen is not None and previous_gen not in ["", " "]:
            if self.is_openai:
                if self.enable_chat:
                    input.append({"role": 'assistant', 'content': previous_gen})
                else:    
                    input += f'<|endoftext|>{previous_gen}'
                
            else:
                input += previous_gen

        return self.truncate_prompt(input)

    def get_string_with_varying_examplars(
        self,
        question,
        retrieval_result=None,
        formatted_reference=None,
        previous_gen=None,
        examplars=[],
        tokenizer=None,
        max_length=2048,
        **params,
    ):
        """
        Select the maximum number of examplars that can be placed in the prompt
        """

        final_examplars = None
        num = len(examplars)
        while len(examplars) > 0:
            for num in range(len(examplars), 0, -1):
                possible_prompt = self.get_string(
                    question=question,
                    retrieval_result=retrieval_result,
                    formatted_reference=formatted_reference,
                    previous_gen=previous_gen,
                    examplars="\n\n".join(examplars[:num]),
                    **params,
                )

                possible_prompt_tokens = tokenizer.encode(possible_prompt)
                if len(possible_prompt_tokens) <= max_length:
                    final_examplars = examplars[:num]
                    break
            if final_examplars is None:
                examplars = examplars[1:]
            else:
                break
        if final_examplars is None:
            final_examplars = []

        final_prompt = self.get_string(
            question=question,
            retrieval_result=retrieval_result,
            formatted_reference=formatted_reference,
            previous_gen=previous_gen,
            examplars="\n\n".join(final_examplars[:num]),
            **params,
        )

        return final_prompt

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            if self.reference_template is not None:
                format_reference += self.reference_template.format(idx=idx, title=title, text=text)
            else:
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
