import scipy.io
import numpy as np
import torch
from layer_config_forward import MultiLayerPerceptron_forward
import math
import scipy.io as sio
from torchsummary import summary
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# Load the forward model
hidden_size = [150,150, 150, 150]
Forward_model = MultiLayerPerceptron_forward(8, hidden_size, 31)
Forward_model.load_state_dict(torch.load('8ink_net_4lay_150.ckpt'))
Forward_model.to(device)
# for param in Forward_model.parameters():
#     param.requires_grad = False


# Load the spectral image to generate the halftone pattern
# mat = scipy.io.loadmat('printer_300_patch_dataset_reflectance.mat')
# test_all=mat['reflectance']
mat = scipy.io.loadmat('gray_spec_gt.mat')
test_all=mat['gray_spec']
def RMSELoss(yhat,y):
    return torch.mean((torch.sqrt(torch.sum((yhat - y)**2,1))))/5.57*100 # RMSE loss

i = 1
Reproduction_error = 0

target_spec = test_all
target_spec_tensor = torch.from_numpy(target_spec)
target_spec_tensor_cuda = target_spec_tensor.to(torch.device("cuda"))
target_spec_tensor_cuda = target_spec_tensor_cuda.type(torch.cuda.FloatTensor)

# approx_halftone = model_backward(test_spec_tensor_cuda)
# Reproduction_error = RMSELoss(Forward_model(approx_halftone), test_spec_tensor_cuda) + Reproduction_error
# mu_x = torch.tensor([0.3199, 0.3193, 0.3150, 0.3138]).to(torch.device("cuda")) # average of training data
R_x = 1


halftone = torch.rand(1,8).to(torch.device("cuda")).type(torch.cuda.FloatTensor).clone().detach().requires_grad_(True)
print(halftone)
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
        optimizer = torch.optim.Adam([halftone], lr=0.001)

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
            #
            # if (i + 1) % 100 == 0:
            #     print(i)
            #     print(' Loss: {:.4f}'
            #           .format(loss.item()))

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
    # time_all.append((time.time() - start_time))
#
# tst = np.array(reproduced_spec_all)
# spec_out = tst[0]
# for i in range(1,901):
#     spec_out = np.concatenate([spec_out, tst[i]], axis = 0)

sio.savemat('backprob_spec_NA_150_4000samples.mat', {'backprob_spec':np.array(reproduced_spec_all)})
sio.savemat('backprob_designs_NA_150_4000samples.mat', {'backprob_designs':np.array(halftone_all)})
sio.savemat('backprob_time_NA_150_4000samples.mat', {'backprob_time':np.array(time_all)})
sio.savemat('backprob_loss_NA_150_4000samples.mat', {'best_loss':np.array(best_loss_all)})

# reproduced_spec = Forward_model(init)
# reproduced_spec = reproduced_spec.to(torch.device("cpu"))
# reproduced_spec = reproduced_spec.detach().numpy()


# path="./grayramp/painting%d" %(i)
# np.save(path,reproduced_spec)
# sio.savemat('reproduced_spec_backprob.mat', {'reproduced_spec_backprob':reproduced_spec})
#
# print('Painting average reproduction RMSE% = {} %'.format(Reproduction_error)
