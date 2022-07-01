function r = get_direction(g, S, Y, H)
    q = g;
    k = size(S, 2);
    alphas = zeros(k, 1);
    rhos = zeros(size(alphas));
    for i = k : -1 : 1
        rhos(i) = 1 / (((Y(:, i)).') * S(:, i));
        alphas(i) = rhos(i) * (((S(:, i)).') * q);
        q = q - alphas(i) * Y(:, i);
    end
    r = H * q;
    for i = 1 : k
        beta_i = rhos(i) * (((Y(:, i)).') * r);
        r = r + (alphas(i) - beta_i) * S(:, i);
    end
end