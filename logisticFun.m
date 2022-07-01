function [f, g, h] = logisticFun(x, A, b, lambda)
    f = 0;
    n = size(A, 1);
    d = size(A, 2);
    g = zeros(d, 1);
%     h = zeros(d);
    
    for i = 1 : n
        inProd = A(i, :) * x;
        M = max(0, inProd);
        eProd = exp(inProd - M);
        
        f = f + log(exp(-M) + eProd) + M - b(i) * inProd;
        g = g + (((A(i, :)).') * (eProd / (exp(-M) + eProd)) - b(i) * ...
            ((A(i, :)).'));
%         h = h + (((A(i, :)).') * ((eProd * exp(-M)) / ...
%             ((exp(-M) + eProd)^2)) * A(i, :));
    end
    [f_reg, g_reg, Hv_reg] = reg(x, lambda);
    f = f + f_reg;
    g = g + g_reg;
    h = @(v) hess_vec(A, x, v) + Hv_reg(v);
end