from copy import deepcopy

def split_dataset(dataset, split_bool:list):
    assert len(split_bool) == len(dataset)
    
    data = dataset.data
    pos_data = [x for x,flag in zip(data,split_bool) if flag]
    neg_data = [x for x,flag in zip(data,split_bool) if not flag]

    pos_dataset, neg_dataset = deepcopy(dataset), deepcopy(dataset)
    pos_dataset.data = pos_data
    neg_dataset.data = neg_data

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
    
    final_dataset = deepcopy(pos_dataset)
    final_dataset.data = final_data
    
    return final_dataset