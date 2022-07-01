function [f_reg, g_reg, Hv_reg] = reg(x, lambda)
    f_reg = (1 / 2) * lambda * norm(x)^2;
    g_reg = lambda * x;
    Hv_reg = @(v) lambda * v;
end