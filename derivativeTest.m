function derivativeTest(fun,x0)
[f0,g0,H0] = fun(x0);
if isa(H0,'function_handle') 
    MVP = true;
else
    MVP = false;
end
dx = randn(size(x0));
M = 20;
dxs = zeros(M,1);
firstOrderError = zeros(M,1);
secondOrderError = zeros(M,1);
for i = 1:M
    x = x0 + dx;
    [f,~] = fun(x);
    firstOrderError(i) = abs(f - ( f0 + (dx')*g0 ));
    if MVP
        secondOrderError(i) = abs(f - ( f0 + (dx')*g0 + 0.5*(dx')*(H0(dx))));
    else
        secondOrderError(i) = abs(f - ( f0 + (dx')*g0 + 0.5*(dx')*(H0*dx)));
    end
    fprintf('First Order Error: %g, Second Order Error: %g\n',firstOrderError(i),secondOrderError(i));
    dxs(i) = norm(dx);
    dx = dx/2;
end

figure(1);
semilogy(1:M,abs(firstOrderError),'r',1:M,dxs.^2,'b');
legend('1st Order Error','Theoretical Order');

figure(2);
semilogy(1:M,abs(secondOrderError),'r',1:M,dxs.^3,'b');
legend('2nd Order Error','Theoretical Order');

end