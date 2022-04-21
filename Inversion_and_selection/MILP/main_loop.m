function [graysample, time, area_coverage] = main_loop(spec_test, l_h, u_h, w_b_net)
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
  
    constraints =[];
    target = spec_test;
    input_size = size(target,1);
    % Ink selection binary variables
    % x_b = binvar(1,lay_neuron_size(1));
    %coreset size
    input_size = size(target,1);
    Z = sdpvar(input_size,31);
    % Define the variables and set the always active and always deactive relus.
    x{1} = sdpvar(input_size,lay_neuron_size(1));

    for lay =2:depth_ind
        x{lay} = sdpvar(input_size,lay_neuron_size(lay),'full');
        z{lay-1} = binvar(input_size,lay_neuron_size(lay),'full');
        out{lay-1} = x{lay-1} * w{lay-1} + repmat(b{lay-1},[input_size 1]);


    end

    % last layer does not have relu!
    y = x{end} * w{end} + repmat(b{end},[input_size 1]);
    % Define the piecewise linear Relu constraints.

    for layer =1:depth_ind-1
        constraints = [constraints,
            x{layer+1} >= out{layer},...
            x{layer+1} >= 0, ...
            x{layer+1} <= repmat(u_h{layer+1},[input_size 1]).*z{layer}, ...
            x{layer+1} <= out{layer}-repmat(l_h{layer+1},[input_size 1]).*(1-z{layer}), ...
            ];
    end

    % Define the selection constraints.
    % Set the unselected inks(from selection step) to zero.
    constraints = [constraints,x{1}([1:31, 33, 35:end])==0 ,0 <= x{1} , x{1} <= 1, Z >= (y-target), Z >= -(y-target)];

    % Define the objective
    objective =   sum(sum(Z));

    options = sdpsettings('solver','gurobi', 'gurobi.Threads', 0,  'gurobi.MIPFocus', 0, 'verbose', 0);
    tic
    sol = optimize(constraints, objective, options);
    time = toc;
    graysample = double(y);
    area_coverage = double(x{1});
    
end