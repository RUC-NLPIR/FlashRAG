import re

class MMPromptTemplate:
    BASE_USER_PROMPT = '{reference}\nBased on the above examples, answer the following question. Only give me the final choices.\nQuestion: {question}\nAnswer: '
    def __init__(self, config, system_prompt=None, user_prompt=None):
        self.config = config
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt if user_prompt is not None else self.BASE_USER_PROMPT
    def get_string(self, item):
        question = item.question if item.question is not None else item.text
        question_image = item.image
        # retrieval_result = item.retrieval_result
        try:
            retrieval_result = item.retrieval_result
        except:
            retrieval_result = []

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        reference_str = ""
        content_list = []
        for idx, item in enumerate(retrieval_result):
            # item is multimodal data or raw text
            if 'image' not in item:
                # raw text item
                reference_str += f'Example {idx+1}: {item["contents"]}\n'
            else:
                content_list.append({'type': 'image', 'image': item['image']})
                reference_str += f'Example {idx+1}: {item["text"]}\n'
        content_list.append({'type': 'image', 'image': question_image})
        content_list.append({'type': 'text', 'text': self.user_prompt.format(question=question, reference=reference_str)})
        messages.append({"role": "user", "content": content_list})
        return messages


class GAOKAOMMPromptTemplate(MMPromptTemplate):
    BASE_USER_PROMPT = "请你做一道{subject}选择题\n请你结合文字和图片一步一步思考,并将思考过程写在【解析】和<eoe>之间。{instruction}\n例如：{example}\n请你严格按照上述格式作答。\n你可以参考一些知识: {reference}。题目如下：{question}"
    INSTRUCTION_DICT = {
        'single_choice': '你将从A，B，C，D等选项中选出正确的答案，并写在【答案】和<eoa>之间。',
        'multiple_choice': '你将从A，B，C，D等选项中选出所有符合题意的答案，并写在【答案】和<eoa>之间。'
    }
    EXAMPLE_DICT = {
        'single_choice': '【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>',
        'multiple_choice': '【答案】 AB <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】... <eoa>'
    }
    def __init__(self, config, system_prompt=None, user_prompt=None):
        self.config = config
        self.system_prompt = system_prompt
        if user_prompt is None:
            self.user_prompt = self.BASE_USER_PROMPT
        else:
            self.user_prompt = user_prompt

    def get_string(self, item):
        question = item.question if item.question is not None else item.text
        question_image = item.image
        question_type = item.question_type
        subject = item.subject
        
        instruction = self.INSTRUCTION_DICT[question_type]
        example = self.EXAMPLE_DICT[question_type]

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        content_list = []
        if '{reference}' not in self.user_prompt:
            user_prompt = self.user_prompt.format(question=question, instruction=instruction, example=example, subject=subject)
        else:
            retrieval_result = item.retrieval_result
            reference_str = ""
            for idx, item in enumerate(retrieval_result):
                # item is multimodal data or raw text
                if 'image' not in item:
                    # raw text item
                    reference_str += f'参考内容 {idx+1}: {item["contents"]}\n'
                else:
                    content_list.append({'type': 'image', 'image': item['image']})
                    reference_str += f'参考内容 {idx+1}: {item["text"]}, 标准答案: {item["golden_answers"][0]}\n'
            user_prompt = self.user_prompt.format(question=question, reference=reference_str, instruction=instruction, example=example, subject=subject)

        content_list.append({'type': 'image', 'image': question_image})
        content_list.append({'type': 'text', 'text': user_prompt})
        messages.append({"role": "user", "content": content_list})
        return messages


class MathVistaPromptTemplate:
    BASE_USER_PROMPT_FREE_FORM = (
        "You are an AI assistant designed to solve mathematical and visual reasoning problems. "
        "Below is a question that requires you to analyze the given information and provide a detailed answer.\n\n"
        "### Reference Information:\n{reference}\n\n"
        "### Question:\n{question}\n\n"
        "### Instructions:\n"
        "Only give me the final answer.\n\n"
        "### Answer:\n"
    )

    BASE_USER_PROMPT_MULTI_CHOICE = (
        "You are an AI assistant designed to solve mathematical and visual reasoning problems. "
        "Below is a multiple-choice question that requires you to analyze the given information and select the correct choice.\n\n"
        "### Reference Information:\n{reference}\n\n"
        "### Question:\n{question}\n\n"
        "### Instructions:\n"
        "Only give me the correct option letter, e.g., A, B, C, D.\n\n"
        "### Answer:\n"
    )

    def __init__(self, config, system_prompt=None, user_prompt_free_form=None, user_prompt_multi_choice=None):
        self.config = config
        self.system_prompt = system_prompt
        self.user_prompt_free_form = user_prompt_free_form if user_prompt_free_form is not None else self.BASE_USER_PROMPT_FREE_FORM
        self.user_prompt_multi_choice = user_prompt_multi_choice if user_prompt_multi_choice is not None else self.BASE_USER_PROMPT_MULTI_CHOICE

    def get_string(self, item, use_retrieval_image=False, use_retrieval_text=False, use_question_image=False):
        question = item.question if item.question is not None else item.text
        question_image = item.image
        question_type = item.question_type  # 'free_form' or 'multi_choice'

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        content_list = []
        reference_str = ""

        # Add retrieval results as reference (if available)
        try:
            retrieval_result = item.retrieval_result
        except:
            retrieval_result = []
            
        for idx, item in enumerate(retrieval_result):
            if 'image' not in item:
                reference_str += f'Example {idx + 1}: {item["contents"]}\n'
            else:
                content_list.append({'type': 'image', 'image': item['image']})
                reference_str += f'Example {idx + 1}: {item["text"]}\n'

        # Add question image (if available)
        if question_image:
            content_list.append({'type': 'image', 'image': question_image})

        # Format the user prompt based on question type
        if question_type == 'free_form':
            user_prompt = self.user_prompt_free_form.format(question=question, reference=reference_str)
        elif question_type == 'multi_choice':
            user_prompt = self.user_prompt_multi_choice.format(question=question, reference=reference_str)
        else:
            raise ValueError(f"Unsupported question type: {question_type}")

        # Add the formatted user prompt
        content_list.append({'type': 'text', 'text': user_prompt})
        messages.append({"role": "user", "content": content_list})

        return messages
    
