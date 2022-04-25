import os

import numpy as np
import scipy.io as sio
import torch

from NN_metamaterial_model import NN_metamaterial

# Parameters
input_size = 8
hidden_size = [500, 500, 500, 500]
output_size = 150


# Model
Forward_model = NN_metamaterial(input_size, hidden_size, output_size)
Forward_model.load_state_dict(torch.load('Metamaterial_4lay_500.ckpt'))
Forward_model.eval()


# Extract parameters
w = list(Forward_model.parameters())
w_numpy = []
for tensor in w:
    w_numpy.append(tensor.detach().numpy())

w_numpy.append(Forward_model.bn1.running_mean.detach().numpy())
w_numpy.append(Forward_model.bn1.running_var.detach().numpy())

w_numpy.append(Forward_model.bn2.running_mean.detach().numpy())
w_numpy.append(Forward_model.bn2.running_var.detach().numpy())

w_numpy.append(Forward_model.bn3.running_mean.detach().numpy())
w_numpy.append(Forward_model.bn3.running_var.detach().numpy())

w_numpy.append(Forward_model.bn4.running_mean.detach().numpy())
w_numpy.append(Forward_model.bn4.running_var.detach().numpy())


# Save parameters
sio.savemat(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'MILP', 'Data', 'NN_metamaterial_4lay_500_w_b.mat'), {'w_b_model':w_numpy})
