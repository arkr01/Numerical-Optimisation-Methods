function hv = hess_vec(A, x, v)
    n = size(A, 1);
    d = size(A, 2);
    hv = zeros(d, 1);
    for i = 1 : n
        inProd = A(i, :) * x;
        M = max(0, inProd);
        eProd = exp(inProd - M);
        hv = hv + ...
            ((((eProd * exp(-M)) / ((exp(-M) + eProd)^2)) * ...
            A(i, :) * v) * ((A(i, :)).'));
    end
end