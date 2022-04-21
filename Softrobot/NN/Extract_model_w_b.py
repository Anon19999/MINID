import numpy as np
import torch
from layer_config_forward import MultiLayerPerceptron_forward
import scipy.io as sio



input_size = 40
hidden_size = [128, 128]
num_classes = 206
# Load the model
Forward_model = MultiLayerPerceptron_forward(input_size, hidden_size, num_classes)
Forward_model.load_state_dict(torch.load('soft_robot_forward_128_128.ckpt'))

# Extract the parameters (weights and biases) from the model.
w = list(Forward_model.parameters())


# Convert pytorch tensor to numpy
w_numpy = []
for tensor in w:
    w_numpy.append(tensor.detach().numpy())

# Save weights and biases as matlab file
# np.save('numpy_w_b', w_numpy)
sio.savemat('soft_robot_w_b_128_128.mat', {'w_numpy':w_numpy})