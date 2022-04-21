import scipy.io
import numpy as np
import torch
from layer_config_forward import MultiLayerPerceptron_forward
import scipy.io as sio
import time
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# Load the forward model
input_size = 44
hidden_size_mu = [50, 50]
num_classes = 31

# Load the model
mu_model = MultiLayerPerceptron_forward(input_size, hidden_size_mu, num_classes)
mu_model.load_state_dict(torch.load( '../44ink_net_50_50.ckpt'))
mu_model.to(device)
mu_model.eval()





i = 1
Reproduction_error = 0
mat = sio.loadmat('spec_map_duotone.mat')
performance_test=mat['spec_map']
target_performance_cuda = torch.tensor(performance_test).to(device).float()


criterion_MSE = nn.MSELoss()
criterion_MSE_no_reduction= nn.MSELoss(reduction = 'none')
#
mu_x = 0
R_x = 0.4
batch_size = target_performance_cuda.shape[0]
n_iter = 2000
reproduced_spec_all = []
design_all = []
start_time = time.time()


best_reproduce_spec = []
best_loss = []
best_design = []
start_time = time.time()
repeats_n = 50
reproduced_mu_many_runs = torch.empty(repeats_n, batch_size, num_classes).to(device)
loss_many_runs = torch.empty(repeats_n, batch_size).to(device)
designs_many_runs = torch.empty(repeats_n, batch_size, input_size).to(device)
batchsize = target_performance_cuda.shape[0]
for repeats in range(0, repeats_n):
    design = (torch.rand(batch_size, input_size)).to(torch.device("cuda")).type(torch.cuda.FloatTensor).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([design], lr=0.02)

    plateau = []
    for i in range(n_iter):
        # print(i)
        optimizer.zero_grad()
        loss = 0
        reproduced_mu = mu_model(design)
        loss_mse = criterion_MSE(reproduced_mu, target_performance_cuda)
        loss = loss_mse
        print(' Loss_mse: {:.4f}'.format(loss_mse.item()))
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #         design.clamp_(-3, 3)
        with torch.no_grad():
            plateau.append(loss.to(torch.device("cpu")).detach().numpy())
        if (i > 10 and np.max(np.abs(np.array(plateau[-10:-1]) - np.array(plateau[-1]))) < 0.000001):
            print('break on iter_%d' % (i))
            print(' Loss: {:.4f}'.format(loss.item()))
            break
        if i > n_iter-2:
            print('iter_%d' % (i))
            print(' Loss: {:.4f}'.format(loss.item()))

    designs_many_runs[repeats,:,:] = design
    reproduced_mu_many_runs[repeats, :, :] = reproduced_mu
    loss_many_runs[repeats, :] = torch.mean(criterion_MSE_no_reduction(reproduced_mu, target_performance_cuda),1)


# save the best convergences for each sample
time_total = time.time() - start_time
values, index = torch.min(loss_many_runs,0)
best_loss = loss_many_runs[index,range(0,loss_many_runs.shape[1])]
best_performance = reproduced_mu_many_runs[index,range(0,loss_many_runs.shape[1]),:]
best_designs = designs_many_runs[index, range(0, loss_many_runs.shape[1]), :]
total_MSE_error = torch.mean((best_performance.to(device) - target_performance_cuda) ** 2)

sio.savemat('Results/NA_performance.mat', {'NA_performance': np.array(best_performance.detach().to(torch.device('cpu')))})
sio.savemat('Results/NA_design.mat', {'NA_design': np.array(best_designs.detach().to(torch.device('cpu')))})
sio.savemat('Results/NA_time.mat', {'NA_time': time_total})
sio.savemat('Results/NA_loss.mat', {'NA_loss': total_MSE_error})



