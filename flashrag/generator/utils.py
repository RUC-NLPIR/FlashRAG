import warnings
import os

def resolve_max_tokens(params: dict, generation_params: dict, prioritize_new_tokens: bool = False) -> dict:
    """
    Resolve and validate max_tokens parameters from both params and generation_params.

    Args:
        params: Dictionary containing user-provided parameters
        generation_params: Dictionary containing generation-specific parameters
        prioritize_new_tokens: If True, max_new_tokens takes precedence over max_tokens
                             If False, max_tokens takes precedence (default behavior)

    Returns:
        Updated generation_params dictionary
    """

    def get_token_params(param_dict: dict) -> tuple:
        """Extract max_tokens and max_new_tokens from a parameter dictionary."""
        return (param_dict.pop("max_tokens", None), param_dict.pop("max_new_tokens", None))

    def resolve_tokens(max_tokens: int, max_new_tokens: int) -> int:
        """
        Resolve between max_tokens and max_new_tokens values based on priority.
        Returns the resolved token value or None if no valid value found.
        """
        # If either value is None, return the non-None value
        if max_tokens is None:
            return max_new_tokens
        if max_new_tokens is None:
            return max_tokens

        # Both values exist but are different
        if max_tokens != max_new_tokens:
            if prioritize_new_tokens:
                warnings.warn(
                    f"max_tokens ({max_tokens}) and max_new_tokens ({max_new_tokens}) "
                    f"are different. Using max_new_tokens value as it has priority."
                )
                return max_new_tokens
            else:
                warnings.warn(
                    f"max_tokens ({max_tokens}) and max_new_tokens ({max_new_tokens}) "
                    f"are different. Using max_tokens value as it has priority."
                )
                return max_tokens

        # Both values are equal
        return max_tokens

    # Try to resolve from params first, then fall back to generation_params
    max_tokens, max_new_tokens = get_token_params(params)
    final_max_tokens = resolve_tokens(max_tokens, max_new_tokens)

    # If no valid tokens found in params, try generation_params
    if final_max_tokens is None:
        max_tokens, max_new_tokens = get_token_params(generation_params)
        final_max_tokens = resolve_tokens(max_tokens, max_new_tokens)

    generation_params.pop("max_new_tokens", None)
    generation_params.pop("max_tokens", None)
    if final_max_tokens is not None:
        if prioritize_new_tokens:
            generation_params["max_new_tokens"] = final_max_tokens
        else:
            generation_params["max_tokens"] = final_max_tokens
    return generation_params


def convert_image_to_base64(image):
    from PIL import Image
    from io import BytesIO
    import base64
    if isinstance(image, Image.Image):
        image = image.convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_bs64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f'data:image/jpeg;base64,{image_bs64}'
    return image

def process_image_pil(image):
    from PIL import Image
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str):
        return load_image_from_source(image)
    else:
        raise ValueError("Image must be a PIL Image or a string path to an image file")

def process_image(content_dict):
    from PIL import Image
    image = content_dict.get('image')
    if isinstance(image, Image.Image):
        content_dict['image'] = convert_image_to_base64(image)
    elif isinstance(image, str):
        content_dict['image'] = convert_image_to_base64(load_image_from_source(image))

def load_image_from_source(image_path):
    from PIL import Image
    import requests
    if os.path.exists(image_path):
        return Image.open(image_path).convert('RGB')
    else:
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        return Image.open(response.raw).convert('RGB')