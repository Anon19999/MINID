clear
%% Load
% Net weights and biases
load('Data/photonic_net_3_100_50_100_w_b.mat')

% Target to be optimized for
load('Data/spects_Si_Tio2_Au_Ag_H2O_test.mat')

%Upper bounds and lower bounds
load('Data/lower_upper_bound_sphere.mat')
%% Test vals

l_h = lower_bound_all;
u_h = upper_bound_all;
w_b_net = w_numpy;

% Determine the size and depth of the net
net_ind = size(w_b_net,2);
depth_ind = net_ind/2;

% Number of neurons in the first layer
lay_neuron_size(1) = size(double(w_b_net{1})',1);

% Initializing the MILP reformulation of the neural network
for lay =1:depth_ind
    w{lay} = double(w_b_net{2*lay-1})';
    b{lay} = double(w_b_net{2*lay});
    lay_neuron_size(lay+1) = size(w{lay},2);
end


time_all_nonint = [];
rounded_design_all = [];
rounded_spec_all=[];
gt_spec_all = [];

n_target=1:6:100;
for n_cnt=1:size(n_target,2)

    constraints =[];
    target = myspects(:,n_target(n_cnt))';

    input_size = size(target,1);

    %coreset size
    input_size = size(target,1);
    Z = sdpvar(input_size,201);
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
    constraints = [constraints, 0 <= x{1} , x{1} <= 1, Z >= (y-target), Z >= -(y-target)];

    % Define the objective
    objective =   sum(sum(Z));

    %Optimization params
    options = sdpsettings('solver','gurobi', 'gurobi.Threads', 0,  'gurobi.MIPFocus', 0);
    tic
    sol = optimize(constraints, objective, options);
    time = toc;
    time_all_nonint(n_cnt) = time;
    rounded_design_all{n_cnt} = round(double(x{1}*7));
    rounded_spec_all{n_cnt} = SimpleNeuralNet(round(double(x{1}*7)/7), w_numpy);
    obj_val_all{n_cnt} = sum(sum(abs(SimpleNeuralNet(round(double(x{1}*7))/7,w_numpy)-target)));
    gt_spec_all{n_cnt} = target;
end
save('Data/rounded_inversion.mat','time_all_nonint','rounded_design_all','rounded_spec_all','obj_val_all','gt_spec_all')
