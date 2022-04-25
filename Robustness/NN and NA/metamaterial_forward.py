import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from NN_metamaterial_model import NN_metamaterial


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
input_size = 8
hidden_size = [500,500, 500, 500]
output_size = 150
num_epochs = 150
batch_size = 20
learning_rate = 3*1e-3
learning_rate_decay = 0.95
weight_decay = 2e-05


# dataset
dataset = TensorDataset(torch.from_numpy(np.load('y_train.npy')).float(), torch.from_numpy(np.load('x_train.npy')).float())
torch.manual_seed(20)
lengths = [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)]
train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)


# model
forward_model = NN_metamaterial(input_size, hidden_size, output_size).to(device)
forward_model.apply(weights_init)
forward_model.to(device)

print(forward_model)

# Loss and optimizer
criterion_MSE = nn.MSELoss()
optimizer = torch.optim.Adam(forward_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train the forward_model
lr = learning_rate
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (Y_data, X_data) in enumerate(train_loader):
        # Move tensors to the configured device
        X_data = X_data.to(device)
        Y_data = Y_data.to(device)
        #################################################################################
        # Implement the training code                                             #
        optimizer.zero_grad()
        outputs = forward_model(X_data)
        loss = 10000*criterion_MSE(outputs, Y_data)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    with torch.no_grad():
        Y_data_all = torch.zeros(0, 300).to(device)
        outputs_all = torch.zeros(0, 300).to(device)
        for Y_data, X_data in val_loader:
            Y_data = Y_data.to(device)
            Y_data_all = torch.cat((Y_data_all,Y_data),0)
            ####################################################
            #evaluation #
            X_data = X_data.to(device)
            outputs = forward_model(X_data)
            outputs = torch.squeeze(outputs)
            outputs_all = torch.cat((outputs_all,outputs),0)

        loss = criterion_MSE(Y_data_all, outputs_all)
        print('Validataion RMSE is: {}'.format(loss))


# save the model
torch.save(forward_model.state_dict(), 'Metamaterial_4lay_500.ckpt')
