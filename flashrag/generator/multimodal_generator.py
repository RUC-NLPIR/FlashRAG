from typing import List
import os
import json
import importlib
from copy import deepcopy
import warnings
import math
from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModel
from abc import abstractmethod
import json
import os
import importlib
import base64
from io import BytesIO
from flashrag.generator.utils import convert_image_to_base64, process_image, resolve_max_tokens, process_image_pil

class BaseMultiModalGenerator:
    """`BaseMultiModalGenerator` is a base object of Generator model."""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.model_path = config["generator_model_path"]

        self.max_input_len = config["generator_max_input_len"]
        self.batch_size = config["generator_batch_size"]
        self.device = config["device"]
        self.gpu_num = torch.cuda.device_count()
        self.config = config
        self.generation_params = config["generation_params"]
    
    def generate(self, input_list: list) -> List[str]:
        """
        input_list: A list contains of messages, each message is a list, like:
        [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image1_path"},
                    {"type": "image", "image": "image2_path"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        
        The content of each dict can be a str(pure text as input) or list(for multimodal input).
        """
        
        pass

class BaseInferenceEngine:
    def __init__(self, model_path, device='cpu', max_input_len=4096):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.max_input_len = max_input_len
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    @torch.inference_mode(mode=True)
    def generate(self, input_list: list, batch_size=None, **params):
        pass

class Qwen2VLInferenceEngine(BaseInferenceEngine):
    def _load_model(self):
        from transformers import Qwen2VLForConditionalGeneration
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True
        ).eval()
        min_pixels = 3136
        max_pixels = 12845056
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor.tokenizer.model_max_length = self.max_input_len
        self.tokenizer = self.processor.tokenizer
    @torch.inference_mode(mode=True)
    def generate(self, input_list, **params):
        # convert image to base64
        for messages in input_list:
            for message in messages:
                if isinstance(message['content'], list):
                    for content_dict in message['content']:
                        if content_dict['type'] == 'image':
                            content_dict['image'] = convert_image_to_base64(content_dict['image'])

        from qwen_vl_utils import process_vision_info
        texts = [self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in input_list]
        image_inputs, video_inputs = process_vision_info(input_list)    
        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        # print(inputs)
        # print(inputs['input_ids'].shape,inputs['attention_mask'].shape,inputs['pixel_values'].shape,inputs['image_grid_thw'].shape)
        outputs = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **params
        )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text

class InternVL2InferenceEngine(BaseInferenceEngine):
    def _load_model(self):
        import torch
        gpu_num = torch.cuda.device_count()
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='auto' if gpu_num <= 1 else self.split_model(),
            trust_remote_code=True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.model_max_length = self.max_input_len
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id 


    def split_model(self):
        # get model name
        with open(os.path.join(self.model_path, 'config.json')) as f:
            config = json.load(f)
        num_layers = config['llm_config']['num_hidden_layers']
        device_map = {}
        world_size = torch.cuda.device_count()
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    def build_transform(self, input_size):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image, input_size=448, max_num=12):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    @torch.inference_mode(mode=True)
    def generate(self, input_list, **params):
        import torch
        # TODO: Currently only support single image batch or multi-image without batch 
        # convert input format to internvl2
        final_prompt_list = [] # each is a str
        final_image_list = [] # each is a list
        system_message = None
        for messages in input_list:
            # parse each query
            for item in messages:
                if item['role'] == 'system':
                    assert isinstance(item['content'], str)
                    system_message = item['content']
                else:
                    if isinstance(item['content'], str):
                        # pure text input
                        final_prompt_list.append(item['content'])
                        final_image_list.append([])
                    else:
                        # multimodal input
                        image_list = [d['image'] for d in item['content'] if d['type'] == 'image']
                        text = [d['text'] for d in item['content'] if d['type'] == 'text'][0]
                        final_prompt_list.append(text)
                        final_image_list.append(image_list)
        if system_message is not None:
            self.model.system_message = system_message
        
        final_image_list = [[self.load_image(img, max_num=12).to(self.model.dtype).to(self.model.device) for img in image_list] for image_list in final_image_list]

        if all([len(image_list) ==1 for image_list in final_image_list]):
            torch.cuda.empty_cache()
            # batch inference with single image
            final_image_list = [image_list[0] for image_list in final_image_list]
            pixel_values = torch.cat(final_image_list, dim=0)
            num_patches_list = [img.size(0) for img in final_image_list]
            final_prompt_list = [f'<image>\n{text}' for text in final_prompt_list]
            outputs = self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                final_prompt_list,
                num_patches_list=num_patches_list,
                generation_config=params,
                history=None,
                return_history=None,
            )
            return outputs
        else:
            # do single item inference with multi image 
            outputs = []
            for image_list, prompt in zip(final_image_list, final_prompt_list):
                torch.cuda.empty_cache()
                pixel_values = torch.cat(image_list, dim=0)
                num_patches_list = [img.size(0) for img in image_list]
                prompt_prefix = ""
                for i in range(len(image_list)):
                    prompt_prefix += f'Image-{i+1}: <image>\n'
                prompt = prompt_prefix + prompt

                output = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    num_patches_list=num_patches_list,
                    generation_config=params,
                    history=None,
                    return_history=None,
                )
                outputs.append(output)
            return outputs

class LlavaInferenceEngine(BaseInferenceEngine):
    def _load_model(self):
        from transformers import AutoProcessor
        with open(os.path.join(self.model_path, "config.json"), "r") as f:
            config = json.load(f)
            model_type = config['architectures'][0]
        model_type = getattr(importlib.import_module('transformers'), model_type)
        self.model = model_type.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = 'left'
        self.tokenizer = self.processor.tokenizer
        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        self.image_token = "<image>"

    @torch.inference_mode(mode=True)
    def generate(self, input_list, **params):
        # add special tokens
        new_input_list = []
        visual_list = []
        for messages in input_list:
            new_messages = []
            for message in messages:
                item_visual_list = []
                if isinstance(message['content'],list):
                    # remove all image
                    image_list = [item['image'] for item in message['content'] if item['type'] == 'image']
                    item_visual_list.extend(image_list)
                    text_content = [item['text'] for item in message['content'] if item['type'] == 'text'][0]
                    image_tokens = " ".join([self.image_token]*len(image_list))
                    text_content = f"{image_tokens}\n{text_content}"
                    new_messages.append({"role": message['role'], "content": text_content})
                else:
                    new_messages.append(message)
                visual_list.append(item_visual_list)
            new_input_list.append(new_messages)
        visual_list = sum(visual_list, [])
        texts = self.tokenizer.apply_chat_template(new_input_list, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=texts, images=visual_list, padding=True, truncation=True, max_length=self.max_input_len, return_tensors='pt').to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **params
        )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        output_text = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return output_text


class HFModelInferenceEngineFactory:
    _engine_map = {
        'qwen': Qwen2VLInferenceEngine,
        'llava': LlavaInferenceEngine,
        'internvl': InternVL2InferenceEngine,
    }

    @staticmethod
    def get_engine(model_path, device='cpu', **kwargs):
        config_file_path = os.path.join(model_path, 'config.json')
        with open(config_file_path, "r") as f:
            model_config = json.load(f)
        model_arch = model_config['architectures'][0]
    
        for engine_name, engine_class in HFModelInferenceEngineFactory._engine_map.items():
            if engine_name in model_arch.lower():
                return engine_class(model_path, device, **kwargs)
            
        raise ValueError(f"Model {model_path} is not supported!")
      
    
class HFMultiModalGenerator(BaseMultiModalGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model_name = config['generator_model']
        self.model_path = config['generator_model_path']
        
        self.inference_engine = HFModelInferenceEngineFactory.get_engine(
            model_path=self.model_path,
            device=self.device,
            max_input_len=self.max_input_len
        )

    @torch.inference_mode(mode=True)
    def generate(
        self,
        input_list: list,
        batch_size=None,
        **params
    ):
        # solve params
        if not isinstance(input_list[0], list):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if 'temperature' not in generation_params:
            generation_params['temperature'] = 0
        if 'do_sample' not in generation_params:
            generation_params['do_sample'] = True if generation_params['temperature'] > 0 else False
        if generation_params['do_sample'] == False:
            generation_params['temperature'] = 0
        # deal stop params
        stop_sym = None
        if "stop" in generation_params:
            from flashrag.generator.stop_word_criteria import StopWordCriteria

            stop_sym = generation_params.pop("stop")
            stopping_criteria = [
                StopWordCriteria(
                    tokenizer=self.tokenizer,
                    prompts=input_list,
                    stop_words=stop_sym,
                )
            ]
            generation_params["stopping_criteria"] = stopping_criteria

        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=True)

        # preprocess input list
        from PIL import Image
        for messages in input_list:
            for message in messages:
                if isinstance(message['content'], list):
                    for content_dict in message['content']:
                        if content_dict['type'] == 'image':
                            content_dict['image'] = process_image_pil(content_dict['image'])

        output_responses = []
        for idx in trange(0, len(input_list), batch_size, desc='Generation process: '):
            torch.cuda.empty_cache()
            batch_prompts = input_list[idx: idx+batch_size]
            output_responses.extend(self.inference_engine.generate(batch_prompts, **generation_params))
        return output_responses









