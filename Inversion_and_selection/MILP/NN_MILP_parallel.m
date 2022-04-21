%% Load
clear

% Target to be optimized for
load('Data/spec_map_duotone.mat')

%Upper bounds and lower bounds
load('Data/lower_upper_bound_Ink_50_50_constraint.mat')

% Net weights and biases
load('Data/44ink_net_50_50_w_b.mat')

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



MILP_map_all = [];
time_all= [];
area_coverage_all = [];
% parpool(64)
parfor i=1:size(spec_map,1)
  spec_test = double(spec_map(i,:));
  MILP_map = [];
  time = [];
  area_coverage = [];
  [MILP_map, time, area_coverage] = main_loop(spec_test, l_h, u_h, w_b_net);
  MILP_map_all = [MILP_map_all ;MILP_map]; 
  time_all = [time_all, time];
  area_coverage_all = [area_coverage_all; area_coverage];
  
end
save('Data/MILP_map_approx_network.mat','time_all','MILP_map_all', 'area_coverage_all')
%%
