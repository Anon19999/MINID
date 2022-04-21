%% Load
clear

if isfile('Data/lower_upper_bound_soft_robot_128_128_zeta_o.mat')
    disp("Bounds are already calculated and can be loaded from file")
    prompt = input("Choose: 1. Recalculate the bounds 2. Load from File\n");
    if prompt == 2
        return
    end
end

load('Data/soft_robot_w_b_128_128.mat')

%% Test vals
% for i=1:10
%     w_numpy{i}(abs(w_numpy{i})<1e-5)=0;
% end
% delete(gcp('nocreate'))
% parpool(128)

tic
for n_layer= 2:size(w_numpy,2)/2
    %
    n_layer
    
    %Load the weights and biases of the neural net.
    w_b_net = w_numpy(1:2*(n_layer));
    % Calculate the tight bounds for big M problrm, if available load them
    % instead.
    % [u_h, l_h] = l_u_calc(w_b_net);
    net_ind = size(w_b_net,2);
    depth_ind = net_ind/2;
    
    
    lower_bound_all{1} = -0.2;
    upper_bound_all{1} = 0.2;
    
    lower_bound = zeros(1,size(w_numpy{2*n_layer-1},2));
    upper_bound = zeros(1,size(w_numpy{2*n_layer-1},2)); 
    parfor n_neuron = 1:size(w_numpy{2*n_layer-1},2)
        n_neuron
        [lower_bound(n_neuron), upper_bound(n_neuron)] = loop_main(n_neuron, lower_bound_all, upper_bound_all, depth_ind, w_b_net);

    end
    
    lower_bound_all{n_layer} = lower_bound;
    upper_bound_all{n_layer} = upper_bound;
    
    
end
time = toc;
save('Data/lower_upper_bound_soft_robot_128_128_zeta_o.2.mat','lower_bound_all','upper_bound_all','time')
