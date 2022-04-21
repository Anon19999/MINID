import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron_forward(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(MultiLayerPerceptron_forward, self).__init__()
        #################################################################################
        # Initialize the modules required to implement the mlp with given layer   #
        # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
        # hidden_layers[-1] --> num_classes                                             #
        #################################################################################
        layers = []
        layers.append(nn.Linear((input_size), (hidden_layers[0])))
        # layers.append(nn.Linear((hidden_layers[0]), (hidden_layers[1])))
        # layers.append(nn.Linear((hidden_layers[1]), (hidden_layers[2])))
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear((hidden_layers[i]), (hidden_layers[i+1])))

        layers.append(nn.Linear((hidden_layers[len(hidden_layers)-1]), (num_classes)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #################################################################################
        # Implement the forward pass computations                                 #
        #################################################################################

        # x = F.relu(self.layers[0](x))
        # x = F.relu(self.layers[1](x))
        # x = F.relu(self.layers[2](x))
        for i in range(len(self.hidden_layers)):
            x = F.relu(self.layers[i](x))
        x = (self.layers[len(self.hidden_layers)](x))
        out=x
        return out