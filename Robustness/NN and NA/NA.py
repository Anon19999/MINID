import time

import numpy as np
import scipy.io as sio
import torch

from NN_metamaterial_model import NN_metamaterial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# Load the forward model
input_size = 8
hidden_size = [500, 500, 500, 500]
output_size = 150
Forward_model = NN_metamaterial(input_size, hidden_size, output_size)
Forward_model.load_state_dict(torch.load('Metamaterial_4lay_500.ckpt'))
Forward_model.eval()
Forward_model.to(device)


def RMSELoss(yhat,y):
    return torch.mean((torch.sqrt(torch.sum((yhat - y)**2,1))))/5.57*100 # RMSE loss

target_performance_tensor = torch.from_numpy(np.load('y_train.npy'))
target_performance_tensor_cuda = target_performance_tensor.to(device)
target_performance_tensor_cuda = target_performance_tensor_cuda.type(torch.cuda.FloatTensor)

mu_x = torch.tensor(np.mean(np.load('x_train.npy'),0)).to(device) # average of training data
R_x = 1


n_iter = 2000
reproduced_performance_all = []
design_all = []
time_all = []
best_loss_all = []
for j in [12000,15000]:
    best_reproduce_performance = []
    best_loss = []
    best_design = []
    start_time = time.time()
    for zz in range(0,50):
        design_data = torch.rand(1, 8).to(torch.device("cuda")).type(torch.cuda.FloatTensor).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([design_data], lr=0.001)

        plateau = []
        for i in range(n_iter):
            optimizer.zero_grad()
            reproduced_performance = Forward_model(design_data)
            loss = torch.abs(reproduced_performance - target_performance_tensor_cuda[j, :]).sum() + torch.relu(torch.abs(design_data - mu_x * R_x) - mu_x * R_x).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                    design_data.clamp_(-1, 1)
            with torch.no_grad():
                plateau.append(loss.to(torch.device("cpu")).detach().numpy())
            if (i>10 and np.max(np.abs(np.array(plateau[-10:-1])-np.array(plateau[-1])))<0.001) :
                print('break on iter_%d' %(i))
                print(' Loss: {:.4f}'.format(loss.item()))
                break

        with torch.no_grad():
            best_reproduce_performance.append(reproduced_performance.to(torch.device("cpu")).detach().numpy())
            loss_ = torch.abs(reproduced_performance - target_performance_tensor_cuda[j, :]).sum()
            best_loss.append(loss_.to(torch.device("cpu")).detach().numpy())
            best_design.append(design_data.to(torch.device("cpu")).detach().numpy())


    for index in np.argsort(np.array(best_loss)):
        if ((best_design[index]<-1).sum() + (best_design[index]>1).sum()) == 0:
            design_all.append(best_design[index])
            reproduced_performance_all.append(best_reproduce_performance[index])
            print ('chosen loss %f, performance_%d' %(np.array(best_loss)[index],j))
            best_loss_all.append(np.array(best_loss)[index])
        elif index:
            print()

    print("--- %s seconds ---" % (time.time() - start_time))
    time_all.append((time.time() - start_time))

sio.savemat('backprob_performance_NA.mat', {'backprob_performance':np.array(reproduced_performance_all)})
sio.savemat('backprob_designs_NA.mat', {'backprob_designs':np.array(design_all)})
sio.savemat('backprob_time_NA.mat', {'backprob_time':np.array(time_all)})
sio.savemat('backprob_loss_NA.mat', {'best_loss':np.array(best_loss_all)})
