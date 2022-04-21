%% Load
clear

load('Data/lower_upper_bound_canon_150s.mat')

load('Data/gray_spec_gt.mat')

load('Data/canon_net_w_b.mat')



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
%Load the target spectra to be reproduced.

graysample_all = [];
time_all= [];
% parpool(128)
parfor i=1:size(gray_spec,1)
    i
    spec_test = double(gray_spec(i,:));
  [graysample, time] = main_loop(spec_test, l_h, u_h, w_b_net);
  graysample_all = [graysample_all ;graysample]; 
  time_all = [time_all, time];
  time
  
end
save('Data/MILP_grayramp.mat','time_all','graysample_all')