load weakly_convex_results.mat
load convex_results.mat
load strongly_convex_results.mat

hold on
plot(1:num_experiments, weakly_convex_results_nonsummable_step, 'Color', 'r');
plot(1:num_experiments, weakly_convex_results_geometric_step, 'Color', 'b');
plot(1:num_experiments, weakly_convex_results_constant_step, 'Color', 'g');
plot(1:length(convex_results), convex_results, 'Color', 'k')
legend('Weakly Convex With Non-summable Step', 'Weakly Convex With Geometric Step', 'Weakly Convex With Constant Step', '$L$-smooth Convex With Non-summable', 'Interpreter', 'latex')
grid on
hold off