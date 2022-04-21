%% Load
% clear

load('Data/backprob_spec_NA.mat')
load('Data/backprob_designs_NA.mat')
load('Data/NN_metamaterial_4lay_500_w_b.mat')
load('Data/lower_upper_bound_metamaterial.mat')
%% Test vals

l_h = lower_bound_all;
u_h = upper_bound_all;
w_b_net = w_b_model;

net_ind = size(w_b_net,2);
depth_ind = 4;

lay_neuron_size(1) = size(double(w_b_net{1})',1);
for lay =1:depth_ind
    w{lay} = double(w_b_net{2*lay-1})';
    b{lay} = double(w_b_net{2*lay});
    gamma{lay} = double(w_b_net{10+2*lay-1});
    beta{lay} = double(w_b_net{10+2*lay});
    mean{lay} = double(w_b_net{26+2*lay-1});
    var{lay} = double(w_b_net{26+2*lay});
    
    lay_neuron_size(lay+1) = size(w{lay},2);
end
    w{5} = double(w_b_net{2*5-1})';
    b{5} = double(w_b_net{2*5});
for i=1:3
    deconv_weight{i} = double(w_b_model{18+2*i-1});
    deconv_bias{i} = double(w_b_model{18+2*i})';
end
conv_weight = double(w_b_model{25});
conv_bias = double(w_b_model{26})';

%% Initializing the variables
%Load the target spectra to be reproduced.

deviation_all = [];
time_all = [];
% let's analyse the top 20 best results out of 50 calculation for two
% different design
n_sol =[1:20, 51:70];
% n_sol = 14:17
for i = 1 : size(n_sol,2)
spec_test =double(squeeze(backprob_spec(n_sol(i),:,:)))';

constraints =[];
target = spec_test;
input_size = size(target,1);
% Ink selection binary variables
% x_b = binvar(1,lay_neuron_size(1));
%coreset size
% input_size = size(target,1);
input_size = 1;
% Z = sdpvar(1,300);
% Define the variables and set the always active and always deactive relus.
x{1} = sdpvar(input_size,lay_neuron_size(1));
design = double(squeeze(backprob_designs(n_sol(i),:,:)))';

for lay =2:depth_ind+1
    x{lay} = sdpvar(input_size,lay_neuron_size(lay),'full');
    z{lay-1} = binvar(input_size,lay_neuron_size(lay),'full');
    out{lay-1} = (((x{lay-1} * w{lay-1} + repmat(b{lay-1},[input_size 1]))-mean{lay-1})./(sqrt(var{lay-1}+1e-5))).*gamma{lay-1}+beta{lay-1};   
end

% last layer does not have relu!
y = x{5} * w{5} + repmat(b{5},[input_size 1]);
y = deconv1D_func(y, deconv_weight{1}, deconv_bias{1}, 3, 2);
y = deconv1D_func(y, deconv_weight{2}, deconv_bias{2}, 2, 1);
y = deconv1D_func(y, deconv_weight{3}, deconv_bias{3}, 2, 1);
y = conv1D_func(y, reshape(conv_weight,[1 4 1]), conv_bias, 0, 1);
% Define the piecewise linear Relu constraints.

for layer =1:depth_ind-1+1
    constraints = [constraints,
        x{layer+1} >= out{layer},...
        x{layer+1} >= 0, ...
        x{layer+1} <= repmat(u_h{layer+1},[input_size 1]).*z{layer}, ...
        x{layer+1} <= out{layer}-repmat(l_h{layer+1},[input_size 1]).*(1-z{layer}), ...
        ];
end

% Define the selection constraints.
% constraints = [constraints, 0 <= x{1} , x{1} <= repmat(x_b,[input_size 1]), sum(x_b)<=12, Z >= (y-target), Z >= -(y-target)];
% constraints = [constraints, -1 <= x{1} , x{1} <= 1,design-1e-5 < x{1}, x{1} < design+1e-5, Z >= (y-target), Z >= -(y-target)];
constraints = [constraints, -1 <= x{1} , x{1} <= 1,design-1e-3 <= x{1}, x{1} <= design+1e-3];


% Define the objective
objective =   -sum(sum(abs((y-target))));
% assign(x{1}, double(squeeze(backprob_inks))')
options = sdpsettings('solver','gurobi', 'gurobi.Threads', 0,  'gurobi.MIPFocus', 0, 'gurobi.LogFile','tst');
tic
sol = optimize(constraints, objective, options);
time = toc;
time_all{i}=time;
deviation_all{i}= -double(objective);

end
save('Data/robustness_results.mat','time_all','deviation_all')
