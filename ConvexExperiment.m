num_experiments = 30;

R = 1;
L = 2;

convex_results = zeros(1,num_experiments);
for N = 1:num_experiments
    
    G_N = sdpvar(N+2);
    F_N = sdpvar(N+1,1);
    
    options = sdpsettings('solver','mosek');
    cost = -1 * F_N(end);
    
    % To avoid implicit rank constraint, we redefine variables in terms of
    % their gradient update – entry ij contains the stepsize at iteration i of
    % gradient j
    selector_x = zeros(N+2,N+2);
    for i = 1:(N+2)
        for j = 1:i
            selector_x(j,i) = -1 / j;
        end
    end
    selector_x(1,:) = -1 * selector_x(1,:);
    
    
    % F_N >= 0 ensures the existence of minimizers (by translation we can
    % assume minimal value is 0)
    % G_N >= 0 is PSD constraint
    % G_N(1,1) <= R is the initial condition
    constraints = [F_N >= 0, G_N >= 0, G_N(1,1) <= R^2];
    for i = 1:(N+1)
        for j = 1:(N+1)
            if(i ~= j)
                I_n = eye(N+2);
                gradient_mat = 1/2 * (I_n(:, j+1)) * (selector_x(:, i) - selector_x(:, j))'...
                    + 1/2 * (selector_x(:, i) - selector_x(:, j)) * (I_n(:, j+1))' ;
                lipschitz_mat = (I_n(:, j+1) - I_n(:, i+1)) * (I_n(:, j+1) - I_n(:, i+1))';
                constraints = [constraints,...
                    F_N(i) >= F_N(j) + trace(gradient_mat * G_N) + 1 / (2 * L) * trace(lipschitz_mat * G_N)];
            end
        end
        % add in minimizer interpolation constraint
        gradient_mat = 1/2 * (I_n(:, i+1)) * (-1 * selector_x(:, i))'...
            + 1/2 * (-1* selector_x(:, i)) * (I_n(:, i+1))' ;
        lipschitz_mat = (-1 * I_n(:, i+1)) * (-1 * I_n(:, i+1))';
        constraints = [constraints, ...
           0 >= F_N(j) + trace(gradient_mat * G_N) + 1 / (2 * L) * trace(lipschitz_mat * G_N)];
    end
    
    diagnostics = optimize(constraints, cost, options);
    
    convex_results(N) = value(F_N(end));
    if(length(convex_results) ~= 30)
        msgbox(sprintf("Wrong length on iteration %d", N));
        error("Wrong length result vector");
        return
    end
end

plot(1:num_experiments, convex_results);
save 'convex_results.mat' convex_results