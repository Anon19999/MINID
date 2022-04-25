import torch
from torch import Tensor


def custom_RMSE_loss(model_output: Tensor, ground_truth: Tensor) -> Tensor:
    return torch.mean((torch.sqrt(torch.sum((ground_truth - model_output)**2,1))))/5.57*100


def custom_sum_mse(model_output: Tensor, ground_truth: Tensor):
    return torch.sum(torch.mean((ground_truth - model_output) ** 2, dim=1))
