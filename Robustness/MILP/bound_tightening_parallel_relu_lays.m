%% Load
clear


if isfile('Data/lower_upper_bound_metamaterial_upto_lay4_4lay_500.mat')
    disp("Bounds are already calculated and can be loaded from file")
    prompt = input("Choose: 1. Recalculate the bounds 2. Load from File\n");
    if prompt == 2
        return
    end
end

load('Data/NN_metamaterial_4lay_500_w_b.mat')

%% Test vals
% for i=1:10
%     w_numpy{i}(abs(w_numpy{i})<1e-3)=0;
% end
w_numpy = w_b_model;
% parpool(64)
tic
for n_layer= 2:4
    %
    n_layer
    
    %Load the weights and biases of the neural net.
    w_b_net = w_numpy(1:2*(n_layer));
    % Calculate the tight bounds for big M problrm, if available load them
    % instead.
    net_ind = size(w_b_net,2);
    depth_ind = net_ind/2;
    
    
    lower_bound_all{1} = -1;
    upper_bound_all{1} = 1;
    
    lower_bound = zeros(1,size(w_numpy{2*n_layer-1},2));
    upper_bound = zeros(1,size(w_numpy{2*n_layer-1},2)); 
    parfor n_neuron = 1:size(w_numpy{2*n_layer-1},2)
        n_neuron
        [lower_bound(n_neuron), upper_bound(n_neuron)] = loop_main2(n_neuron, lower_bound_all, upper_bound_all, depth_ind, w_b_model);

    end
    
    lower_bound_all{n_layer} = lower_bound;
    upper_bound_all{n_layer} = upper_bound;
    
    
end
time = toc
save('Data/lower_upper_bound_metamaterial_upto_lay4_4lay_500.mat','lower_bound_all','upper_bound_all', 'time')
