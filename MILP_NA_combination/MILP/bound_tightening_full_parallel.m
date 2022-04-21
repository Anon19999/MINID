%% Load
clear

if isfile('Data/lower_upper_bound_8ink_net_4lay_150.mat')
    disp("Bounds are already calculated and can be loaded from file")
    prompt = input("Choose: 1. Recalculate the bounds 2. Load from File\n");
    if prompt == 2
        return
    end
end

load('8ink_net_4lay_150_w_b.mat')

%% Test vals

for n_layer= 2:size(w_numpy,2)/2

    %Load the weights and biases of the neural net.
    w_b_net = w_numpy(1:2*(n_layer));
    % Calculate the tight bounds for big M problrm, if available load them
    % instead.
    % [u_h, l_h] = l_u_calc(w_b_net);
    net_ind = size(w_b_net,2);
    depth_ind = net_ind/2;
    
    
    lower_bound_all{1} = 0;
    upper_bound_all{1} = 1;
    
    lower_bound = zeros(1,size(w_numpy{2*n_layer-1},2));
    upper_bound = zeros(1,size(w_numpy{2*n_layer-1},2)); 
    parfor n_neuron = 1:size(w_numpy{2*n_layer-1},2)
        n_neuron
        [lower_bound(n_neuron), upper_bound(n_neuron)] = loop_main(n_neuron, lower_bound_all, upper_bound_all, depth_ind, w_b_net);

    end
    
    lower_bound_all{n_layer} = lower_bound;
    upper_bound_all{n_layer} = upper_bound;
    
    
end
save('lower_upper_bound_8ink_net_4lay_150.mat','lower_bound_all','upper_bound_all')
