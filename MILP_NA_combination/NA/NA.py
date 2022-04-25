import time

import numpy as np
import scipy.io
import scipy.io as sio
import torch

from layer_config_forward import MultiLayerPerceptron_forward

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# Load the forward model
input_size = 8
hidden_size = [150, 150, 150, 150]
output_size = 31
learning_rate = 0.001
Forward_model = MultiLayerPerceptron_forward(input_size, hidden_size, output_size)
Forward_model.load_state_dict(torch.load('8ink_net_4lay_150.ckpt'))
Forward_model.to(device)

target_spec_tensor = torch.from_numpy(scipy.io.loadmat('../MILP/Data/gray_spec_gt.mat')['gray_spec'])
target_spec_tensor_cuda = target_spec_tensor.to(torch.device("cuda"))
target_spec_tensor_cuda = target_spec_tensor_cuda.type(torch.cuda.FloatTensor)

R_x = 1

n_iter = 2000
reproduced_spec_all = []
halftone_all = []
time_all = []
best_loss_all = []
for j in range(100,500,100):
    best_reproduce_spec = []
    best_loss = []
    best_halftone = []
    start_time = time.time()
    time_sample = []
    for zz in range(0,4000):
        halftone = torch.rand(1, 8).to(torch.device("cuda")).type(torch.cuda.FloatTensor).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([halftone], lr=learning_rate)

        plateau = []
        for i in range(n_iter):
            optimizer.zero_grad()
            reproduced_spec = Forward_model(halftone)
            loss = torch.abs(reproduced_spec - target_spec_tensor_cuda[j, :]).sum() + torch.relu(torch.abs(halftone-0.5*R_x) - 0.5*R_x).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                    halftone.clamp_(0, 1)
            with torch.no_grad():
                plateau.append(loss.to(torch.device("cpu")).detach().numpy())
            if (i>10 and np.max(np.abs(np.array(plateau[-10:-1])-np.array(plateau[-1])))<0.0001) :
                print('break on iter_%d' %(i))
                print(' Loss: {:.4f}'.format(loss.item()))
                break
            if i>1998:
                print(' Loss: {:.4f}'.format(loss.item()))

        with torch.no_grad():
            time_sample.append((time.time() - start_time))
            best_reproduce_spec.append(reproduced_spec.to(torch.device("cpu")).detach().numpy())
            loss_ = torch.abs(reproduced_spec - target_spec_tensor_cuda[j, :]).sum()
            best_loss.append(loss_.to(torch.device("cpu")).detach().numpy())
            best_halftone.append(halftone.to(torch.device("cpu")).detach().numpy())


    for index in np.argsort(np.array(time_sample)):
        if ((best_halftone[index]<0).sum() + (best_halftone[index]>1).sum()) == 0:
            halftone_all.append(best_halftone[index])
            reproduced_spec_all.append(best_reproduce_spec[index])
            print ('chosen loss %f, spec_%d' %(np.array(best_loss)[index],j))
            best_loss_all.append(np.array(best_loss)[index])
            time_all.append(time_sample[index])
            # break
        elif index:
            print()

    print("--- %s seconds ---" % (time.time() - start_time))

sio.savemat('backprob_spec_NA_150_4000samples.mat', {'backprob_spec':np.array(reproduced_spec_all)})
sio.savemat('backprob_designs_NA_150_4000samples.mat', {'backprob_designs':np.array(halftone_all)})
sio.savemat('backprob_time_NA_150_4000samples.mat', {'backprob_time':np.array(time_all)})
sio.savemat('backprob_loss_NA_150_4000samples.mat', {'best_loss':np.array(best_loss_all)})
