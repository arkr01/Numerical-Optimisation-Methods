function [f, X, gn, i, t] = newtonCG(theta, beta1, tol, k, x0, A, b, ...
    lambda, cg_k)
    X = zeros(size(A, 2), k + 1);
    gn = zeros(k, 1);
    f = zeros(size(gn));
    t = zeros(size(gn));
    X(:, 1) = x0;
    alpha_max = 1;
    tic;
    for i = 1 : k
        [f(i), g, hv] = logisticFun(X(:, i), A, b, lambda);
        gn(i) = norm(g);
        if gn(i) <= tol
            gn = gn(1 : i);
            t(i) = toc;
            break;
        end
        p = pcg(hv, -g, theta, cg_k);
        alpha_i = armijo(X(:, i), A, b, lambda, alpha_max, beta1, p);
        X(:, i + 1) = X(:, i) + alpha_i * p;
        t(i) = toc;
    end
    X = X(:, 1 : i + 1);
    t = t(1 : i);
    f = f(1 : i);
end