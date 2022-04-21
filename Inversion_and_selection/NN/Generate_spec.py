import numpy as np
import torch
from layer_config_forward import MultiLayerPerceptron_forward
import scipy.io as sio




# Load the forward model
hidden_size = [50,50]
Forward_model = MultiLayerPerceptron_forward(4, hidden_size, 31)
Forward_model.load_state_dict(torch.load('Duotone_net_50_50.ckpt'))

mat = sio.loadmat('area_coverage_green_blue.mat')
HalftoneData=mat['area_coverage_green_blue']
x_train_tensor = torch.from_numpy(HalftoneData).float()

map_spec_duotone_net = Forward_model(x_train_tensor)


sio.savemat('map_spec_duotone_net.mat', {'map_spec_duotone_net':map_spec_duotone_net.detach().numpy()})