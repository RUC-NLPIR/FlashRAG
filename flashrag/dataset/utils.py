from flashrag.dataset import Dataset

def filter_dataset(dataset: Dataset, filter_func = None):
    if filter_func is None:
        return dataset
    data = dataset.data
    for item in data:
        if not filter_func(item):
            data.remove(item)
    return Dataset(config=dataset.config, data=data)

def split_dataset(dataset: Dataset, split_bool:list):
    assert len(split_bool) == len(dataset)

    data = dataset.data
    pos_data = [x for x,flag in zip(data,split_bool) if flag]
    neg_data = [x for x,flag in zip(data,split_bool) if not flag]

    pos_dataset = Dataset(config=dataset.config, data=pos_data)
    neg_dataset = Dataset(config=dataset.config, data=neg_data)

    return pos_dataset, neg_dataset

def merge_dataset(pos_dataset: Dataset, neg_dataset: Dataset, merge_bool: list):
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

def get_batch_dataset(dataset: Dataset, batch_size=16):
    data = dataset.data
    for idx in range(0, len(data), batch_size):
        batched_data = data[idx:idx+batch_size]
        batch_dataset = Dataset(config=dataset.config, data=batched_data)
        yield batch_dataset

def merge_batch_dataset(dataset_list: Dataset):
    dataset = dataset_list[0]
    total_data = []
    for batch_dataset in dataset_list:
        total_data.extend(batch_dataset.data)
    dataset = Dataset(config=dataset.config, data=total_data)
    return dataset