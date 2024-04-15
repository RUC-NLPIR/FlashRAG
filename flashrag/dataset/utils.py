from flashrag.dataset import Dataset
from copy import deepcopy

def split_dataset(dataset, split_bool:list):
    assert len(split_bool) == len(dataset)
    
    data = dataset.data
    pos_data = [x for x,flag in zip(data,split_bool) if flag]
    neg_data = [x for x,flag in zip(data,split_bool) if not flag]

    pos_dataset = Dataset(config=dataset.config, data=pos_data)
    neg_dataset = Dataset(config=dataset.config, data=neg_data)

    return pos_dataset, neg_dataset

def merge_dataset(pos_dataset, neg_dataset, merge_bool: list):
    assert (len(merge_bool) == (len(pos_dataset) + len(neg_dataset)))

    pos_data_iter = iter(pos_dataset.data)
    neg_data_iter = iter(neg_dataset.data)
    
    final_data = []
    
    for is_pos in merge_bool:
        if is_pos:
            final_data.append(next(pos_data_iter))
        else:
            final_data.append(next(neg_data_iter))
    
    final_dataset = Dataset(config=pos_dataset.config, data=final_data)
    
    return final_dataset

def get_batch_dataset(dataset, batch_size=16):
    data = dataset.data
    for idx in range(0, len(data), batch_size):
        batched_data = data[idx:idx+batch_size]
        batch_dataset = Dataset(config=dataset.config, data=batched_data)
        yield batch_dataset

def merge_batch_dataset(dataset_list):
    data = [batch_dataset.data for batch_dataset in dataset_list]
    dataset = Dataset(config=dataset.config, data=data)
    return dataset