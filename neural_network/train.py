from typing import Callable

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from neural_network import settings, utils, test


def _train_forward_network_one_epoch(model: Module, criterion: Callable[[Tensor, Tensor], Tensor], optimizer: Optimizer, train_dataset_loader: DataLoader, epoch_number: int, number_of_epochs: int) -> None:
    model.train()
    for batch_number, (data, ground_truth) in enumerate(train_dataset_loader, 1):
        data = data.to(settings.DEVICE)
        ground_truth = ground_truth.to(settings.DEVICE)

        model_output = model(data)
        loss = criterion(model_output, ground_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_number % settings.NUMBER_OF_BATCH_PER_LOG == 0:
            print(f'Epoch [{epoch_number}/{number_of_epochs}], Batch [{batch_number}/{len(train_dataset_loader)}]', f'Loss: {loss.item():.4f}')
    print()


def train_forward_network(model: Module, criterion: Callable[[Tensor, Tensor], Tensor], optimizer: Optimizer, learning_rate: float, learning_rate_decay: float, train_dataset_loader: DataLoader, test_dataset_loader: DataLoader, number_of_epochs: int) -> None:
    print("Training Forward-Network ...")
    model.train()
    for epoch_number in range(1, number_of_epochs + 1, 1):
        _train_forward_network_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_dataset_loader=train_dataset_loader,
            epoch_number=epoch_number,
            number_of_epochs=number_of_epochs
        )
        learning_rate *= learning_rate_decay
        utils.custom_update_learning_rate(optimizer=optimizer, learning_rate=learning_rate)
        test.test_forward_network(
            model=model,
            criterion=criterion,
            test_dataset_loader=test_dataset_loader
        )
    print("Forward-Network Trained.")
    print()
    print()
