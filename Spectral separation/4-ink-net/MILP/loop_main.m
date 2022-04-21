function [lower_bound, upper_bound] = loop_main(n_neuron, lower_bound, upper_bound, depth_ind, w_b_net)
        
        lay_neuron_size(1) = size(double(w_b_net{1})',1);
        for lay =1:depth_ind
            w{lay} = double(w_b_net{2*lay-1})';
            b{lay} = double(w_b_net{2*lay});
            lay_neuron_size(lay+1) = size(w{lay},2);
        end

        %% Initializing the variables

        %coreset size
        input_size = 1; 

        % Define the variables and set the always active and always deactive relus.
        x{1} = sdpvar(input_size,lay_neuron_size(1));
        for lay =2:depth_ind-1
            x{lay} = sdpvar(input_size,lay_neuron_size(lay));
            z{lay-1} = binvar(input_size,lay_neuron_size(lay));
            out{lay-1} = x{lay-1} * w{lay-1} + repmat(b{lay-1},[input_size 1]);
        end
        x{depth_ind} = sdpvar(1,1);
        z{depth_ind-1} = binvar(1,1);
        out{depth_ind-1} = x{depth_ind-1} * w{depth_ind-1}(:,n_neuron) + b{depth_ind-1}(n_neuron);

        % Define the piecewise linear Relu constraints.
        constraints =[];
        for lay =1:depth_ind-2
            constraints = [constraints,
                x{lay+1} >= out{lay},...
                x{lay+1} >= 0, ...
                x{lay+1} <= repmat(upper_bound{lay+1},[input_size 1]).*z{lay}, ...
                x{lay+1} <= out{lay}-repmat(lower_bound{lay+1},[input_size 1]).*(1-z{lay}), ...
                ];
        end

        % Define the selection constraints.
        constraints = [constraints, 0 <= x{1} , x{1} <= 1];

        % Define the objective
        objective =   out{depth_ind-1};
        options = sdpsettings('solver','gurobi', 'gurobi.IntFeasTol', 1e-9, 'gurobi.Threads', 0, 'gurobi.TimeLimit', 150, 'savesolveroutput',1);
        sol = optimize(constraints, objective, options);
        lower_bound = double(sol.solveroutput.result.objbound)-1e-6;

        % Define the objective
        objective =   -out{depth_ind-1};
        options = sdpsettings('solver','gurobi', 'gurobi.IntFeasTol', 1e-9, 'gurobi.Threads', 0, 'gurobi.TimeLimit', 150, 'savesolveroutput',1);
        sol = optimize(constraints, objective, options);
        upper_bound = -double(sol.solveroutput.result.objbound)+1e-6;
 end