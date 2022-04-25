from typing import List

import torch.nn as nn
from torch.nn import Module


def create_linear_sequential_model(layers_size: List[int]) -> Module:
    layers = []

    for i in range(0, len(layers_size)-1):
        layers.append(nn.Linear(in_features=layers_size[i], out_features=layers_size[i+1], bias=True))
        if i < len(layers_size)-2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)
