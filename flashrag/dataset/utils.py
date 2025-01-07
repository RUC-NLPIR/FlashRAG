from typing import Dict, Any, Union
import numpy as np
from flashrag.dataset import Dataset


def convert_numpy(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: convert_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy(element) for element in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer,)):
        return int(data)
    elif isinstance(data, (np.floating,)):
        return float(data)
    elif isinstance(data, (np.bool_)):
        return bool(data)
    elif isinstance(data, (np.str_)):
        return str(data)
    else:
        return data

def filter_dataset(dataset: Dataset, filter_func=None):
    if filter_func is None:
        return dataset
    data = dataset.data
    for item in data:
        if not filter_func(item):
            data.remove(item)
    return Dataset(config=dataset.config, data=data)


def split_dataset(dataset: Dataset, split_symbol: list):
    assert len(split_symbol) == len(dataset)

    data = dataset.data
    data_split = {symbol: [] for symbol in set(split_symbol)}
    for symbol in set(split_symbol):
        symbol_data = [x for x, x_symbol in zip(data, split_symbol) if x_symbol == symbol]
        data_split[symbol] = Dataset(config=dataset.config, data=symbol_data)

    return data_split


def merge_dataset(dataset_split: dict, split_symbol: list):
    assert len(split_symbol) == sum([len(data) for data in dataset_split.values()])
    dataset_split_iter = {symbol: iter(dataset.data) for symbol, dataset in dataset_split.items()}

    final_data = []
    for item_symbol in split_symbol:
        final_data.append(next(dataset_split_iter[item_symbol]))
    final_dataset = Dataset(config=list(dataset_split.values())[0].config, data=final_data)

    return final_dataset


def get_batch_dataset(dataset: Dataset, batch_size=16):
    data = dataset.data
    for idx in range(0, len(data), batch_size):
        batched_data = data[idx : idx + batch_size]
        batch_dataset = Dataset(config=dataset.config, data=batched_data)
        yield batch_dataset


def merge_batch_dataset(dataset_list: Dataset):
    dataset = dataset_list[0]
    total_data = []
    for batch_dataset in dataset_list:
        total_data.extend(batch_dataset.data)
    dataset = Dataset(config=dataset.config, data=total_data)
    return dataset
def remove_images(data: Any) -> Any:
    from PIL import Image
    from typing import Any
    if isinstance(data, dict):
        return {key: remove_images(value) 
                for key, value in data.items()
                if not isinstance(value, Image.Image)}
    elif isinstance(data, list):
        return [remove_images(element) 
                for element in data 
                if not isinstance(element, Image.Image)]
    elif isinstance(data, tuple):
        return tuple(remove_images(element) 
                     for element in data 
                     if not isinstance(element, Image.Image))
    elif isinstance(data, set):
        return {remove_images(element) 
                for element in data 
                if not isinstance(element, Image.Image)}
    else:
        return data


def clean_prompt_image(input):
    try:
        for message in input:
            if isinstance(message.get("content"), list):
                message["content"] = [item for item in message["content"] if item.get("type") != "image"]
        return input
    except:
        return input