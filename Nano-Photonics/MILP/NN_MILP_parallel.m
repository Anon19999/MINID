%% Load
clear

%Upper bounds and lower bounds
load('lower_upper_bound_canon_150s_1e-5w_removed.mat')


%% Test vals
for i=1:10
    w_numpy{i}(abs(w_numpy{i})<1e-5)=0;
end
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


graysample_all = [];
time_all= [];
for i=1:size(gray_spec,1)
  spec_test = double(gray_spec(i,:));
  [graysample, time] = main_loop(spec_test, l_h, u_h, w_b_net);
  graysample_all = [graysample_all ;graysample]; 
  time_all = [time_all, time];
  time
end
save('Data/MILP_grayramp.mat','time_all','graysample_all')