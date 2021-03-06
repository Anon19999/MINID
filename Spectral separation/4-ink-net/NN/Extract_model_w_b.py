import numpy as np
import torch
from layer_config_forward import MultiLayerPerceptron_forward
import scipy.io as sio



# Load the forward model
hidden_size = [100,100, 100, 100]
Forward_model = MultiLayerPerceptron_forward(4, hidden_size, 31)
Forward_model.load_state_dict(torch.load('canon_net.ckpt'))

# Extract the parameters (weights and biases) from the model.
w = list(Forward_model.parameters())


# Convert pytorch tensor to numpy
w_numpy = []
for tensor in w:
    w_numpy.append(tensor.detach().numpy())

# Save weights and biases as matlab file
# np.save('numpy_w_b', w_numpy)
sio.savemat('../MILP/canon_net_w_b.mat', {'w_numpy':w_numpy})