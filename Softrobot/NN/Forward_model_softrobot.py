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
input_size = 40
hidden_size = [128, 128]
num_classes = 206
num_epochs = 200
# batch_size = 200
learning_rate = 1*1e-3
learning_rate_decay = 0.98
reg = 0.001


# mat = scipy.io.loadmat('dataset_toy.mat')
# stateData = mat['y_dataset']
# stateData = stateData.T
#
# mat = scipy.io.loadmat('dataset_toy.mat')
# controllData = mat['x_dataset']
# controllData = controllData.T

Design_data = np.empty((0,40))
Prformance_data = np.empty((0,103,2))
for data_n in range(5):
    path_name = 'data/soft-robot/input_%d.npy' %(data_n+1)
    Design_data = np.append(Design_data, np.load(path_name), axis=0)
    path_name = 'data/soft-robot/output_%d.npy' % (data_n + 1)
    Prformance_data = np.append(Prformance_data, np.load(path_name), axis=0)





x_train_tensor = torch.from_numpy(Design_data).float()
y_train_tensor = torch.from_numpy(Prformance_data).float()
y_train_tensor = torch.reshape(y_train_tensor, (y_train_tensor.size(0), -1))


dataset = TensorDataset(x_train_tensor, y_train_tensor)

torch.manual_seed(data_n)
lengths = [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)]

train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=8)


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
            # x = F.relu(self.layers[i](x))
            x = self.layers[i](x)
            x = torch.relu(x)
        x = (self.layers[len(hidden_size)](x))
        out=x
        return out

model_forward = MultiLayerPerceptron_forward(input_size, hidden_size, num_classes).to(device)

model_forward.apply(weights_init)
model_forward.to(device)

# Loss and optimizer
def sum_mse(yhat,y):
    return torch.sum(torch.mean((yhat - y) ** 2, dim=1))

criterion_MSE = nn.MSELoss()
criterion_sum_mse = sum_mse
optimizer = torch.optim.Adam(model_forward.parameters(), lr=learning_rate)

# Train the model_forward
lr = learning_rate
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (controll, state) in enumerate(train_loader):
        # Move tensors to the configured device
        controll = controll.to(device)
        state = state.to(device)
        #################################################################################
        # Implement the training code                                             #
        optimizer.zero_grad()
        # im = controll.view(31, input_size)
        outputs = model_forward(controll)
        loss = criterion_sum_mse(outputs, state)
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
        state_all = torch.zeros(0, num_classes).to(device)
        outputs_all = torch.zeros(0, num_classes).to(device)
        for controll, state in val_loader:
            state = state.to(device)
            state_all = torch.cat((state_all,state),0)
            ####################################################
            #evaluation #
            controll = controll.to(device)
            # outputs = model_forward(controll.view(31, input_size))
            outputs = model_forward(controll)
            outputs_all = torch.cat((outputs_all,outputs),0)

        loss = criterion_MSE(state_all, outputs_all)
        # loss = ((lab_color_gt - lab_color_output)**2).mean(axis=None)
        print('Validataion RMSE is: {}'.format(loss))


# save the model
torch.save(model_forward.state_dict(), 'soft_robot_forward_128_128.ckpt')




