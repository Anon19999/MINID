import torch.nn as nn
import torch.nn.functional as F


class NN_metamaterial(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(NN_metamaterial, self).__init__()
        #################################################################################
        # Initialize the modules required to implement the mlp with given layer   #
        # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
        # hidden_layers[-1] --> num_classes                                             #
        #################################################################################
        layers = []
        layers.append(nn.Linear((input_size), (hidden_layers[0])))
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear((hidden_layers[i]), (hidden_layers[i + 1])))

        layers.append(nn.Linear((hidden_layers[len(hidden_layers) - 1]), (num_classes)))
        self.layers = nn.Sequential(*layers)

        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.bn3 = nn.BatchNorm1d(hidden_layers[2])
        self.bn4 = nn.BatchNorm1d(hidden_layers[3])
        # self.bn5 = nn.BatchNorm1d(150)

        self.convtrans1 = nn.ConvTranspose1d(1, 4, 8, 2, 3)
        self.convtrans2 = nn.ConvTranspose1d(4, 4, 5, 1, 2)
        self.convtrans3 = nn.ConvTranspose1d(4, 4, 5, 1, 2)
        self.conv1 = nn.Conv1d(4, 1, 1, 1)

    def forward(self, x):
        #################################################################################
        # Implement the forward pass computations                                 #
        #################################################################################

        x = F.relu(self.bn1(self.layers[0](x)))
        x = F.relu(self.bn2(self.layers[1](x)))
        x = F.relu(self.bn3(self.layers[2](x)))
        x = F.relu(self.bn4(self.layers[3](x)))
        x = (self.layers[4](x))

        x = x.unsqueeze(1)

        x = self.convtrans1(x)
        x = self.convtrans2(x)
        x = self.convtrans3(x)
        x = self.conv1(x)
        x = x.squeeze(1)
        return x