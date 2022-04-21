function F = SimpleNeuralNet(x,w_numpy)
% Initialise random weights
w_1 = double(w_numpy{1})';
w_2 = double(w_numpy{3})';
w_3 = double(w_numpy{5})';


b_1 =   double(w_numpy{2});
b_2 = double(w_numpy{4});
b_3 = double(w_numpy{6});


% n_neuron_lay_input = size(w_1,1);
% n_neuron_lay_2 = size(w_1,2);
% n_neuron_lay_3 = size(w_2,2);
% n_neuron_lay_4 = size(w_3,2);

% example input
% x = [0.754686681982361,0.276025076998578,0.679702676853675];
% Forward pass
F = max(0, max(0, x * w_1 + b_1) * w_2 + b_2) * w_3 + b_3;