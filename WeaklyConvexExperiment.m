num_experiments = 30;

rho = 1;
R = 1;
M = 1;

weakly_convex_results_nonsummable_step = zeros(1,num_experiments);
weakly_convex_results_geometric_step = zeros(1,num_experiments);
weakly_convex_results_constant_step = zeros(1,num_experiments);
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
    constraints = [F_N >= 0, G_N >= 0, G_N(1,1) <= R^2, G_N(2:end, 2:end) <= M^2 * eye(N+1)];
    for i = 1:(N+1)
        for j = 1:(N+1)
            if(i ~= j)
                x_i_mat = selector_x(:, i) * selector_x(:, i)';
                x_j_mat = selector_x(:, j) * selector_x(:, j)';
                I_n = eye(N+2);
                gradient_mat = 1/2 * (I_n(:, j+1) + rho * selector_x(:, j)) * (selector_x(:, i) - selector_x(:, j))'...
                    + 1/2 * (selector_x(:, i) - selector_x(:, j)) * (I_n(:, j+1) + rho * selector_x(:, j))' ;
                constraints = [constraints,...
                    F_N(i) + rho / 2 * trace(x_i_mat * G_N) >= F_N(j) + rho / 2 * trace(x_j_mat * G_N) + trace(gradient_mat * G_N)];
            end
        end
        % add in minimizer interpolation constraint
        gradient_mat = 1/2 * (I_n(:, i+1) + rho * selector_x(:, i)) * (-1 * selector_x(:, i))'...
            + 1/2 * (-1* selector_x(:, i)) * (I_n(:, i+1) + rho * selector_x(:, i))' ;
        constraints = [constraints, ...
           0 >= F_N(j) + rho / 2 * trace(x_i_mat * G_N) + trace(gradient_mat * G_N)];
    end
    
    diagnostics = optimize(constraints, cost, options);
    
    weakly_convex_results_nonsummable_step(N) = value(F_N(end));
    if(length(weakly_convex_results_nonsummable_step) ~= 30)
        msgbox(sprintf("Wrong length on iteration %d", N));
        error("Wrong length result vector");
        return
    end
end

for N = 1:num_experiments
    
    G_N = sdpvar(N+2);
    F_N = sdpvar(N+1,1);
    
    options = sdpsettings('solver','mosek');
    cost = -1 * F_N(end);
    
    % To avoid implicit rank constraint, we redefine variables in terms of
    % their gradient update – entry ij contains the stepsize at iteration i of
    % gradient j
    selector_x = ones(N+2,N+2) * -100;
    selector_x = tril(selector_x)';
    for i = 1:(N+2)
        for j = 1:i
            selector_x(j,i) = selector_x(j,i)*(0.8)^(j-1);
        end
    end
    selector_x(1,:) = ones(size(selector_x(1,:)));
    
    
    % F_N >= 0 ensures the existence of minimizers (by translation we can
    % assume minimal value is 0)
    % G_N >= 0 is PSD constraint
    % G_N(1,1) <= R is the initial condition
    constraints = [F_N >= 0, G_N >= 0, G_N(1,1) <= R^2, G_N(2:end, 2:end) <= M^2 * eye(N+1)];
    for i = 1:(N+1)
        for j = 1:(N+1)
            if(i ~= j)
                x_i_mat = selector_x(:, i) * selector_x(:, i)';
                x_j_mat = selector_x(:, j) * selector_x(:, j)';
                I_n = eye(N+2);
                gradient_mat = 1/2 * (I_n(:, j+1) + rho * selector_x(:, j)) * (selector_x(:, i) - selector_x(:, j))'...
                    + 1/2 * (selector_x(:, i) - selector_x(:, j)) * (I_n(:, j+1) + rho * selector_x(:, j))' ;
                constraints = [constraints,...
                    F_N(i) + rho / 2 * trace(x_i_mat * G_N) >= F_N(j) + rho / 2 * trace(x_j_mat * G_N) + trace(gradient_mat * G_N)];
            end
        end
        % add in minimizer interpolation constraint
        gradient_mat = 1/2 * (I_n(:, i+1) + rho * selector_x(:, i)) * (-1 * selector_x(:, i))'...
            + 1/2 * (-1* selector_x(:, i)) * (I_n(:, i+1) + rho * selector_x(:, i))' ;
        constraints = [constraints, ...
           0 >= F_N(j) + rho / 2 * trace(x_i_mat * G_N) + trace(gradient_mat * G_N)];
    end
    
    diagnostics = optimize(constraints, cost, options);
    
    weakly_convex_results_geometric_step(N) = value(F_N(end));
    if(length(weakly_convex_results_geometric_step) ~= 30)
        msgbox(sprintf("Wrong length on iteration %d", N));
        error("Wrong length result vector");
        return
    end
end


for N = 1:num_experiments
    
    G_N = sdpvar(N+2);
    F_N = sdpvar(N+1,1);
    
    options = sdpsettings('solver','mosek');
    cost = -1 * F_N(end);
    
    % To avoid implicit rank constraint, we redefine variables in terms of
    % their gradient update – entry ij contains the stepsize at iteration i of
    % gradient j
    selector_x = ones(N+2,N+2) * -1;
    selector_x = tril(selector_x)';
    selector_x(1,:) = ones(size(selector_x(1,:)));
    
    
    % F_N >= 0 ensures the existence of minimizers (by translation we can
    % assume minimal value is 0)
    % G_N >= 0 is PSD constraint
    % G_N(1,1) <= R is the initial condition
    constraints = [F_N >= 0, G_N >= 0, G_N(1,1) <= R^2, G_N(2:end, 2:end) <= M^2 * eye(N+1)];
    for i = 1:(N+1)
        for j = 1:(N+1)
            if(i ~= j)
                x_i_mat = selector_x(:, i) * selector_x(:, i)';
                x_j_mat = selector_x(:, j) * selector_x(:, j)';
                I_n = eye(N+2);
                gradient_mat = 1/2 * (I_n(:, j+1) + rho * selector_x(:, j)) * (selector_x(:, i) - selector_x(:, j))'...
                    + 1/2 * (selector_x(:, i) - selector_x(:, j)) * (I_n(:, j+1) + rho * selector_x(:, j))' ;
                constraints = [constraints,...
                    F_N(i) + rho / 2 * trace(x_i_mat * G_N) >= F_N(j) + rho / 2 * trace(x_j_mat * G_N) + trace(gradient_mat * G_N)];
            end
        end
        % add in minimizer interpolation constraint
        gradient_mat = 1/2 * (I_n(:, i+1) + rho * selector_x(:, i)) * (-1 * selector_x(:, i))'...
            + 1/2 * (-1* selector_x(:, i)) * (I_n(:, i+1) + rho * selector_x(:, i))' ;
        constraints = [constraints, ...
           0 >= F_N(j) + rho / 2 * trace(x_i_mat * G_N) + trace(gradient_mat * G_N)];
    end
    
    diagnostics = optimize(constraints, cost, options);
    
    weakly_convex_results_constant_step(N) = value(F_N(end));
    if(length(weakly_convex_results_constant_step) ~= 30)
        msgbox(sprintf("Wrong length on iteration %d", N));
        error("Wrong length result vector");
        return
    end
end

hold on
plot(1:num_experiments, weakly_convex_results_nonsummable_step, 'Color', 'r');
plot(1:num_experiments, weakly_convex_results_geometric_step, 'Color', 'b');
plot(1:num_experiments, weakly_convex_results_constant_step, 'Color', 'g');
legend('Non-summable Step', 'Geometric Step', 'Constant Step', 'Interpreter', 'latex')
grid on
hold off
save 'weakly_convex_results.mat' weakly_convex_results_nonsummable_step weakly_convex_results_geometric_step weakly_convex_results_constant_step