%% Load
clear

load('Data/gray_spec_gt.mat')
load('Data/lower_upper_bound_Ink_50_50.mat')
load('Data/44ink_net_50_50_w_b.mat')

%% Test vals

l_h = lower_bound_all;
u_h = upper_bound_all;

%Load the weights and biases of the neural net.
w_b_net = w_numpy;
% Calculate the tight bounds for big M problrm, if available load them
% instead.

net_ind = size(w_b_net,2);
depth_ind = net_ind/2;

lay_neuron_size(1) = size(double(w_b_net{1})',1);
for lay =1:depth_ind
    w{lay} = double(w_b_net{2*lay-1})';
    
    b{lay} = double(w_b_net{2*lay});
    
    lay_neuron_size(lay+1) = size(w{lay},2);
end


%% Initializing the variables

MILP_map_all = [];
time_all= [];
area_coverage_all = [];

n_samples = 1:1:size(gray_spec,1);
parfor i=1:size(n_samples,2)
    i
  spec_test = double(gray_spec(n_samples(i),:));
  MILP_map = [];
  time = [];
  area_coverage = [];
  [MILP_map, time, area_coverage] = main_loop(spec_test, l_h, u_h, w_b_net);
  MILP_map_all = [MILP_map_all ;MILP_map]; 
  time_all = [time_all, time];
  area_coverage_all = [area_coverage_all; area_coverage];
  
end
save('Data/MILP_gray_ramp_50_50.mat','time_all','MILP_map_all', 'area_coverage_all')