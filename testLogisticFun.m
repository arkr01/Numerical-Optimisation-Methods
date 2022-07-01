clear; clc;
n = 1000;
d = 100;
A = randn(n, d);
I = eye(2, 1);
ind = randsample(2, n, true);
b = I(ind, :);
lambda = 1;
derivativeTest(@(x) logisticFun(x, A, b, lambda), ones(d, 1));