clear
addpath('functions')
%% Load
%Load the weights and biases of the neural net.
load('Data/44ink_net_50_50_w_b.mat')

% Load Core Set
load('Data/coreset_6.mat')

%Upper bounds and lower bounds
load('Data/lower_upper_bound_Ink_50_50_constraint.mat')
%% Test vals

l_h = lower_bound_all;
u_h = upper_bound_all;
w_b_net = w_numpy;

% Determine the size and depth of the net
net_ind = size(w_b_net,2);
depth_ind = net_ind/2;

% Number of neurons in the first layer
lay_neuron_size(1) = size(double(w_b_net{1})',1);

% Initialize the MILP reformulation of the neural network
for lay =1:depth_ind
    w{lay} = double(w_b_net{2*lay-1})';
    b{lay} = double(w_b_net{2*lay});
    lay_neuron_size(lay+1) = size(w{lay},2);
end


spec_test = double(selected_coreset);


constraints =[];
target = spec_test;
input_size = size(target,1);

% Ink selection binary variables
x_b = binvar(1,lay_neuron_size(1));

%coreset size
input_size = size(target,1);
Z = sdpvar(input_size,31);
% Define the variables and set the always active and always deactive relus.
x{1} = sdpvar(input_size,lay_neuron_size(1));

% Initializing the MILP reformulation of the neural network
for lay =2:depth_ind
    x{lay} = sdpvar(input_size,lay_neuron_size(lay),'full');
    z{lay-1} = binvar(input_size,lay_neuron_size(lay),'full');
    out{lay-1} = x{lay-1} * w{lay-1} + repmat(b{lay-1},[input_size 1]);    
end

% last layer does not have relu!
y = x{end} * w{end} + repmat(b{end},[input_size 1]);

% Define the piecewise linear Relu constraints.
for layer =1:depth_ind-1
    constraints = [constraints,
        x{layer+1} >= out{layer},...
        x{layer+1} >= 0, ...
        x{layer+1} <= repmat(u_h{layer+1},[input_size 1]).*z{layer}, ...
        x{layer+1} <= out{layer}-repmat(l_h{layer+1},[input_size 1]).*(1-z{layer}), ...
        ];
end

% Define the selection constraints.
constraints = [constraints, 0 <= x{1} , x{1} <= repmat(x_b,[input_size 1]), sum(x_b)<=2, Z >= (y-target), Z >= -(y-target)];

% Define the objective
objective =   sum(sum(Z));

%Optimization params
options = sdpsettings('solver','gurobi', 'gurobi.Threads', 80,  'gurobi.MIPFocus', 0);
tic
sol = optimize(constraints, objective, options);
time = toc
selected_inks = double(x_b);
save('Data/ink_selection_6_inks.mat','time','selected_inks')
double(x{1}) 
