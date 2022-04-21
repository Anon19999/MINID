%% Load
clear
load('Data/lower_upper_bound_soft_robot_128_128_zeta_o.2.mat')
load('Data/soft_robot_w_b_128_128.mat')
load('Data/Mesh_points.mat')
load('Data/obstacles_1_100_50000.mat')
load('Data/Performances_all.mat');
load('Data/Designs_all.mat')
designs = double(Designs_all);
state_params = double(Performances_all);
% sample_num = 1:100:50000;



%% Test vals
% for i=1:10
%     w_numpy{i}(abs(w_numpy{i})<1e-5)=0;
% end
l_h = lower_bound_all;
u_h = upper_bound_all;
% for i=1:4
%   l_h{i} = lower_bound{i}-1e-16;
%   u_h{i} = upper_bound{i}+1e-16;
% end

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
%Load the target spectra to be reproduced.

% spec_test = double(gray_spec(i,:));

design_all = [];
time_all= [];
%calculate the results for 50 random samples
rng(5)
random_samples = randperm(50000,1000);
% delete(gcp('nocreate'))
% parpool(128)
vertex_center = 2:3:63;
vertex_all = 2:63;
vertex_edge = setdiff(vertex_all,vertex_center);
for i = 1:1000
    i
  target = state_params(random_samples(i),:);
  target = target(:,123:124);
  obstacles_ = double(obstacles(random_samples(i),:));
  [design, time] = main_loop(target, l_h, u_h, w_b_net, obstacles_, Mesh_points,vertex_edge);
  design_all = [design_all ;design]; 
  time_all = [time_all, time];
  time
  save('Data/softrobot_withObstacle_zeta_0.2.mat','time_all','design_all')
end

