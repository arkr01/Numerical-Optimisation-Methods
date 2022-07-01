function alpha_i = armijo(x, A, b, lambda, alpha_max, beta1, p)
    i = 0;
    rho = 1 / 2;
    alpha_i = (rho ^ i) * alpha_max;
    [f, g, ~] = logisticFun(x, A, b, lambda);
    [f_next, ~, ~] = logisticFun(x + alpha_i * p, A, b, lambda);
    inProd = (g.') * p;
    while f_next > f + alpha_i * beta1 * inProd
        i = i + 1;
        alpha_i = (rho ^ i) * alpha_max;
        [f_next, ~, ~] = logisticFun(x + alpha_i * p, A, b, lambda);
    end
end