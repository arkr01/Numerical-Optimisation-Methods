function [f, X, gn, i, t] = grad_descent(beta1, tol, k, x0, A, b, lambda)
    X = zeros(size(A, 2), k + 1);
    gn = zeros(k, 1);
    t = zeros(size(gn));
    f = zeros(size(gn));
    X(:, 1) = x0;
    Lg = (1 / 4) * norm(A)^2 + lambda;
    alpha_max = 10 / Lg;
    tic;
    for i = 1 : k
        [f(i), g, ~] = logisticFun(X(:, i), A, b, lambda);
        gn(i) = norm(g);
        if gn(i) <= tol
            gn = gn(1 : i);
            t(i) = toc;
            break;
        end
        p = -g;
        alpha_i = armijo(X(:, i), A, b, lambda, alpha_max, beta1, p);
        X(:, i + 1) = X(:, i) + alpha_i * p;
        t(i) = toc;
    end
    X = X(:, 1 : i + 1);
    f = f(1 : i);
    t = t(1 : i);
end