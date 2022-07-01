function [f_reg, g_reg, Hv_reg] = reg1(x)
    f_reg = (1 / 2) * norm(x)^2;
    g_reg = x;
    Hv_reg = @(v)v;
end