from typing import Tuple

import scipy.io
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_tensor_from_mat_file(path: str, key: str) -> Tensor:
    return torch.from_numpy(scipy.io.loadmat(path)[key]).float()


def create_tensor_dataset(*tensors: Tensor) -> TensorDataset:
    return TensorDataset(*tensors)


def get_training_testing_tensor_datasets(tensor_dataset: TensorDataset, train_split_ratio: float) -> Tuple[TensorDataset, TensorDataset]:
    training_size = int(len(tensor_dataset) * train_split_ratio)
    testing_size = len(tensor_dataset) - training_size
    training_tensor_dataset, testing_tensor_dataset = random_split(tensor_dataset, [training_size, testing_size])
    return training_tensor_dataset, testing_tensor_dataset


def create_data_loader(tensor_dataset: TensorDataset, batch_size: int, shuffle: bool, pin_memory: bool) -> DataLoader:
    return DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
