import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
# from siren import SIREN
import scipy.io

import numpy as np


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 4
hidden_size = [100, 50, 100]
num_classes = 201
num_epochs = 400
# batch_size = 200
learning_rate = 3*1e-3
learning_rate_decay = 0.99
reg=0.0001





mat = scipy.io.loadmat('spects_Si_Tio2_Au_Ag_H2O.mat')
SpecData=mat['myspects']
SpecData = SpecData.T

mat = scipy.io.loadmat('values_Si_Tio2_Au_Ag_H2O.mat')
HalftoneData=mat['values']
HalftoneData = HalftoneData/70



x_train_tensor = torch.from_numpy(SpecData).float()
y_train_tensor = torch.from_numpy(HalftoneData).float()

dataset = TensorDataset(x_train_tensor, y_train_tensor)

# torch.manual_seed(0)
lengths = [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)]

train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=31*3, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=31,shuffle=True)


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
        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear((hidden_layers[i]), (hidden_layers[i+1])))

        layers.append(nn.Linear((hidden_layers[len(hidden_size)-1]), (num_classes)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #################################################################################
        # Implement the forward pass computations                                 #
        #################################################################################

        # x = F.relu(self.layers[0](x))
        # x = F.relu(self.layers[1](x))
        # x = F.relu(self.layers[2](x))
        for i in range(len(hidden_size)):
            x = F.relu(self.layers[i](x))
        x = (self.layers[len(hidden_size)](x))
        out=x
        return out

model_HalftoneTospec = MultiLayerPerceptron_forward(input_size, hidden_size, num_classes).to(device)

model_HalftoneTospec.apply(weights_init)
model_HalftoneTospec.to(device)

# Loss and optimizer
def RMSELoss(yhat,y):
    return torch.mean((torch.sqrt(torch.sum((yhat - y)**2,1))))/14.1774*100

criterion_MSE = nn.MSELoss()
criterion_RMSE = RMSELoss
optimizer = torch.optim.Adam(model_HalftoneTospec.parameters(), lr=learning_rate)

# Train the model_HalftoneTospec
lr = learning_rate
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (spec, halftone) in enumerate(train_loader):
        # Move tensors to the configured device
        halftone = halftone.to(device)
        spec = spec.to(device)
        #################################################################################
        # Implement the training code                                             #
        optimizer.zero_grad()
        # im = halftone.view(31, input_size)
        outputs = model_HalftoneTospec(halftone)
        loss = 1000*criterion_MSE(outputs, spec)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    with torch.no_grad():
        correct = 0
        total = 0
        spec_all = torch.zeros(0, num_classes).to(device)
        outputs_all = torch.zeros(0, num_classes).to(device)
        for spec, halftone in val_loader:
            spec = spec.to(device)
            spec_all = torch.cat((spec_all,spec),0)
            ####################################################
            #evaluation #
            halftone = halftone.to(device)
            # outputs = model_HalftoneTospec(halftone.view(31, input_size))
            outputs = model_HalftoneTospec(halftone)
            outputs_all = torch.cat((outputs_all,outputs),0)

        loss = criterion_MSE(spec_all, outputs_all)
        # loss = ((lab_color_gt - lab_color_output)**2).mean(axis=None)
        print('Validataion MSE is: {}'.format(loss))


# save the model
torch.save(model_HalftoneTospec.state_dict(), 'photonic_net_smallertst.ckpt')




