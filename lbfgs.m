function [f, X, gn, iter, t] = lbfgs(m, beta1, beta2, tol, k, x0, A, b, ...
    lambda)
    d = size(A, 2);
    X = zeros(d, k + 1);
    gn = zeros(k, 1);
    t = zeros(size(gn));
    f = zeros(size(gn));
    X(:, 1) = x0;
    alpha_max = 1;
    i = 1;
    iter = 1;
    tic;
    while iter < k
        [f(iter), g, ~] = logisticFun(X(:, iter), A, b, lambda);
        gn(iter) = norm(g);
        if gn(iter) <= tol
            gn = gn(1 : iter);
            t(iter) = toc;
            break;
        end
        if i == 1
            p = -g;
        else
            H = ((((S(:, i - 1)).') * (Y(:, i - 1))) / ...
                (((Y(:, i - 1)).') * (Y(:, i - 1)))) * eye(d);
            p = -get_direction(g, S, Y, H);
        end
        [alpha_i, ~] = lineSearchWolfeStrong(@(v)softMaxFun(v, A, b, ...
            @reg1), X(:, iter), p, alpha_max, beta1, beta2, k);
        X(:, iter + 1) = X(:, iter) + alpha_i * p;
        if i == m + 1
            S = S(:, 2 : end);
            Y = Y(:, 2 : end);
        end
        i = min(i, m); % cap i at m once it reaches m
        S(:, i) = X(:, iter + 1) - X(:, iter);
        [~, g_next, ~] = logisticFun(X(:, iter + 1), A, b, lambda);
        Y(:, i) = g_next - g;
        iter = iter + 1;
        i = i + 1;
        t(iter) = toc;
    end
    X = X(:, 1 : iter);
    t = t(1 : iter);
    f = f(1 : iter);
end