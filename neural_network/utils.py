import random

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer


def custom_init_weights_and_biases(module: Module) -> None:
    if type(module) == nn.Linear:
        module.weight.data.normal_(0.0, 1e-3)
        module.bias.data.fill_(0.0)


def custom_update_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def save_model(model: Module, path: str) -> None:
    print("Saving Model ...")
    torch.save(model.state_dict(), path)
    print("Model Saved.")
    print()
    print()


def load_model(model: Module, path: str) -> Module:
    print("Loading Model ...")
    model.load_state_dict(torch.load(path))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    print("Model Loaded.")
    print()
    print()
    return model


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True


def save_model_weights_and_biases(model: Module, path: str, key: str) -> None:
    parameters = list(model.parameters())
    parameters_numpy = []

    for tensor in parameters:
        parameters_numpy.append(tensor.detach().numpy())

    scipy.io.savemat(path, {key: parameters_numpy})
