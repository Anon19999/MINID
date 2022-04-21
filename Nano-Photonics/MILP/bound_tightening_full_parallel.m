clear
%% Load
load('Data/photonic_net_3_100_50_100_w_b.mat')

if isfile('Data/lower_upper_bound_sphere.mat')
    disp("Bounds are already calculated and can be loaded from file")
    prompt = input("Choose: 1. Recalculate the bounds 2. Load from File\n");
    if prompt == 2
        return
    end
end
%% The optimization loop
parpool(128)
tic
for n_layer= 2:size(w_numpy,2)/2
    % Define the subnet for bound computation
    w_b_net = w_numpy(1:2*(n_layer));
    % Determine the size and depth of the subnets
    net_ind = size(w_b_net,2);
    depth_ind = net_ind/2;
    
    % Define the range of the inputs
    lower_bound_all{1} = 0;
    upper_bound_all{1} = 1;
    
    % Create placeholder  
    lower_bound = zeros(1,size(w_numpy{2*n_layer-1},2));
    upper_bound = zeros(1,size(w_numpy{2*n_layer-1},2)); 
    
    % Parallel for loop
    parfor n_neuron = 1:size(w_numpy{2*n_layer-1},2)
        n_neuron
        [lower_bound(n_neuron), upper_bound(n_neuron)] = loop_main(n_neuron, lower_bound_all, upper_bound_all, depth_ind, w_b_net);

    end
    
    % Stack calculated bounds
    lower_bound_all{n_layer} = lower_bound;
    upper_bound_all{n_layer} = upper_bound;
    
    
end
time = toc
save('Data/lower_upper_bound_sphere.mat','lower_bound_all','upper_bound_all','time')
delete(gcp('nocreate'))

 


