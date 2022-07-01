clear; clf; hold off; close all; clc;% rng('default');

k = 1000;
beta1 = 10^(-4);
beta2 = 0.9;
m = 20;
tol = beta1;
lambda = 1;
theta = 10^(-2);
cg_k = 100;

[A_train, b_train, A_test, b_test] = loadData;
x0 = zeros(size(A_train, 2), 1);

[objValg, Xg, gng, ig, tg] = grad_descent(beta1, tol, k, x0, A_train, ...
    b_train, lambda);
[objValn, Xn, gnn, in, tn] = newtonCG(theta, beta1, tol, k, x0, ...
    A_train, b_train, lambda, cg_k);
[objVall, Xl, gnl, il, tl] = lbfgs(m, beta1, beta2, tol, k, x0, ...
    A_train, b_train, lambda);

accuracy_g = zeros(ig, 1);
accuracy_n = zeros(in, 1);
accuracy_l = zeros(il, 1);

n = size(A_test, 1);
for i = 1 : ig
    labels = zeros(n, 1);
    for j = 1 : n
        labels(j) = classifyD(A_test(j, :).', Xg(:, i));
    end
    accuracy_g(i) = sum(labels == b_test(:, 1)) / length(b_test);
end

for i = 1 : in
    labels = zeros(n, 1);
    for j = 1 : n
        labels(j) = classifyD(A_test(j, :).', Xn(:, i));
    end
    accuracy_n(i) = sum(labels == b_test(:, 1)) / length(b_test);
end

for i = 1 : il
    labels = zeros(n, 1);
    for j = 1 : n
        labels(j) = classifyD(A_test(j, :).', Xl(:, i));
    end
    accuracy_l(i) = sum(labels == b_test(:, 1)) / length(b_test);
end

ig = 1 : ig;
in = 1 : in;
il = 1 : il;

hold on;
semilogy(ig, objValg, 'r');
semilogy(in, objValn, 'b');
semilogy(il, objVall, 'k');
xlabel('Number of Iterations');
ylabel('Objective Value');
title('Comparison of Optimisation Methods: Objective Value vs # Iterations');
legend({'Gradient Descent', 'Newton-CG', 'L-BFGS'}, 'Location', 'southeast');
print('q2f1.jpg', '-djpeg');

hold off;
figure;
semilogy(ig, gng, 'r');
hold on;
semilogy(in, gnn, 'b');
semilogy(il, gnl, 'k');
xlabel('Number of Iterations');
ylabel('Gradient Norm');
title('Comparison of Optimisation Methods: Gradient Norm vs # Iterations');
legend({'Gradient Descent', 'Newton-CG', 'L-BFGS'}, 'Location', 'southeast');
print('q2f2.jpg', '-djpeg');

hold off;
figure;
semilogy(ig, accuracy_g, 'r');
hold on;
semilogy(in, accuracy_n, 'b');
semilogy(il, accuracy_l, 'k');
xlabel('Number of Iterations');
ylabel('Test Accuracy');
title('Comparison of Optimisation Methods: Test Accuracy vs # Iterations');
legend({'Gradient Descent', 'Newton-CG', 'L-BFGS'}, 'Location', 'southeast');
print('q2f3.jpg', '-djpeg');

hold off;
figure;
semilogy(tg, objValg, 'r');
hold on;
semilogy(tn, objValn, 'b');
semilogy(tl, objVall, 'k');
xlabel('Wall-Clock Time (seconds)');
ylabel('Objective Value');
title('Comparison of Optimisation Methods: Objective Value vs Time (s)');
legend({'Gradient Descent', 'Newton-CG', 'L-BFGS'}, 'Location', 'southeast');
print('q2f4.jpg', '-djpeg');

hold off;
figure;
semilogy(tg, gng, 'r');
hold on;
semilogy(tn, gnn, 'b');
semilogy(tl, gnl, 'k');
xlabel('Wall-Clock Time (seconds)');
ylabel('Gradient Norm');
title('Comparison of Optimisation Methods: Gradient Norm vs Time (s)');
legend({'Gradient Descent', 'Newton-CG', 'L-BFGS'}, 'Location', 'southeast');
print('q2f5.jpg', '-djpeg');

hold off;
figure;
semilogy(tg, accuracy_g, 'r');
hold on;
semilogy(tn, accuracy_n, 'b');
semilogy(tl, accuracy_l, 'k');
xlabel('Wall-Clock Time (seconds)');
ylabel('Test Accuracy');
title('Comparison of Optimisation Methods: Test Accuracy vs Time (s)');
legend({'Gradient Descent', 'Newton-CG', 'L-BFGS'}, 'Location', 'southeast');
print('q2f6.jpg', '-djpeg');