import scipy.io
import numpy as np
import torch
from layer_config_forward import MultiLayerPerceptron_forward
import math
import scipy.io as sio


def spec_gen(halftone):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: %s' % device)

    # Load the forward model
    # Load the forward model
    Forward_model = MultiLayerPerceptron_forward(8, [50, 50, 50], 31)
    Forward_model.load_state_dict(torch.load('Relu_spec.ckpt'))
    Forward_model.to(device)
    for param in Forward_model.parameters():
        param.requires_grad = False



    test_spec = halftone
    test_spec_tensor = torch.from_numpy(test_spec)
    test_spec_tensor_cuda = test_spec_tensor.to(torch.device("cuda"))
    test_spec_tensor_cuda = test_spec_tensor_cuda.type(torch.cuda.FloatTensor)



    reproduced_spec = Forward_model(test_spec_tensor_cuda)
    reproduced_spec = reproduced_spec.to(torch.device("cpu"))
    reproduced_spec = reproduced_spec.detach().numpy()
    return reproduced_spec

# data = np.array([[0.1609,    0.0732 ,   0.0838  ,  0.0189  ,  0.1047   , 0.1524  ,  0.0375  ,  0.1424]])
# print(spec_gen(data))