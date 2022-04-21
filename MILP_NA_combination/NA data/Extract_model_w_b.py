import numpy as np
import torch
from layer_config_forward import MultiLayerPerceptron_forward
import scipy.io as sio



# Load the forward model
hidden_size = [150,150, 150, 150]
Forward_model = MultiLayerPerceptron_forward(8, hidden_size, 31)
Forward_model.load_state_dict(torch.load('8ink_net_4lay_150.ckpt'))

# Extract the parameters (weights and biases) from the model.
w = list(Forward_model.parameters())


# Convert pytorch tensor to numpy
w_numpy = []
for tensor in w:
    w_numpy.append(tensor.detach().numpy())

# Save weights and biases as matlab file
# np.save('numpy_w_b', w_numpy)
sio.savemat('8ink_net_4lay_150_w_b.mat', {'w_numpy':w_numpy})