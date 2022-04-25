from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from neural_network import settings


def test_forward_network(model: Module, criterion: Callable[[Tensor, Tensor], Tensor], test_dataset_loader: DataLoader) -> None:
    print("Testing Forward-Network ...")
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for data, ground_truth in test_dataset_loader:
            data = data.to(settings.DEVICE)
            ground_truth = ground_truth.to(settings.DEVICE)

            model_output = model(data)
            loss = criterion(model_output, ground_truth)

            total_loss += loss.item()

        print(f'Loss: {total_loss / len(test_dataset_loader):.4f}')
    model.train()
    print("Forward-Network Tested.")
    print()
    print()
