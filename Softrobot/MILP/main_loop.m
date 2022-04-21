
function [design, time] = main_loop(target, l_h, u_h, w_b_net, obstacles_, Mesh_points,vertex_edge)
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
    target = target;
    input_size = size(target,1);
    % Ink selection binary variables
    % x_b = binvar(1,lay_neuron_size(1));
    %coreset size
    input_size = size(target,1);
    Z = sdpvar(input_size,size(target,2));
    % Define the variables and set the always active and always deactive relus.
    x{1} = sdpvar(input_size,lay_neuron_size(1));
        vector1 = x{1}(:, 1:20);
    vector2 = x{1}(:, 21:end);

    for lay =2:depth_ind
        x{lay} = sdpvar(input_size,lay_neuron_size(lay),'full');
        z{lay-1} = binvar(input_size,lay_neuron_size(lay),'full');
        out{lay-1} = x{lay-1} * w{lay-1} + repmat(b{lay-1},[input_size 1]);

    end

    % last layer does not have relu!
    y = x{end} * w{end} + repmat(b{end},[input_size 1]);
    target_reproduced = y(:,123:124);
    % Define the piecewise linear Relu constraints.
    constraints = [];
    for layer =1:depth_ind-1
        constraints = [constraints,
            x{layer+1} >= out{layer},...
            x{layer+1} >= 0, ...
            x{layer+1} <= repmat(u_h{layer+1},[input_size 1]).*z{layer}, ...
            x{layer+1} <= out{layer}-repmat(l_h{layer+1},[input_size 1]).*(1-z{layer}), ...
            ];
    end
    % define obstacle loss

    Meshpoints = double(Mesh_points);
    finalpose = y + Meshpoints;
    obstacles_ = repmat(obstacles_,1,103);
    finalpose_obstecle_dist = abs(finalpose - obstacles_);
    finalpose_obstecle_dist = finalpose_obstecle_dist(1:2:end) + finalpose_obstecle_dist(2:2:end);
    
    r=0.9;
    % Define the selection constraints.
    % constraints = [constraints, 0 <= x{1} , x{1} <= repmat(x_b,[input_size 1]), sum(x_b)<=2, Z >= (y-target), Z >= -(y-target)];
    constraints = [constraints, -0.2 <= x{1} , x{1} <= 0.2, Z >= (target_reproduced-target), Z >= -(target_reproduced-target)];
    constraints = [constraints, -0.2<=(vector1(:,2:end-1)-(vector1(:,3:end) + vector1(:,1:end-2))), (vector1(:,2:end-1)-(vector1(:,3:end) + vector1(:,1:end-2)))<=0.2];
    constraints = [constraints, -0.2<=(vector2(:,2:end-1)-(vector2(:,3:end) + vector2(:,1:end-2))), (vector2(:,2:end-1)-(vector2(:,3:end) + vector2(:,1:end-2)))<=0.2];
    constraints = [constraints, finalpose_obstecle_dist(vertex_edge(10:30)) >=r*sqrt(2)];
    % Define the objective
    objective =   sum(sum(Z));

    options = sdpsettings('solver','gurobi', 'gurobi.Threads', 0,  'gurobi.MIPFocus', 0, 'gurobi.TimeLimit', 1000);
    sol = optimize(constraints, objective, options);
    time = sol.solvertime;
    design = double(x{1});
    
end