function [alpha,itrLS] = lineSearchWolfeStrong(objFun, xk, pk, alpha0, c1, c2, linesearchMaxItrs)
[fk, gk] = objFun(xk);
phi_0 = fk;
d_phi_0 = dot(pk,gk);
phi_alpha_prev = phi_0;
alpha_prev = 0;
alpha = alpha0;
itrLS = 0;
zoomFun = @(alpha_low,alpha_high,itrLS)zoom(objFun, xk, fk, gk, pk, c1, c2, linesearchMaxItrs, alpha_low,alpha_high,itrLS);

while itrLS < linesearchMaxItrs
    [phi_alpha , g_alpha] = objFun(xk+alpha*pk);
    d_phi_alpha = dot(pk,g_alpha);
    if ( phi_alpha > phi_0 + c1*alpha*d_phi_0 ||  ( ( phi_alpha  >= phi_alpha_prev ) && itrLS > 0 ) )
        [alpha,itrZoom] = zoomFun(alpha_prev,alpha,itrLS);
        break;
    end
    if abs(d_phi_alpha) <= -c2*d_phi_0
        itrZoom = 0;
        break;
    end
    if d_phi_alpha >= 0
        [alpha,itrZoom] = zoomFun(alpha,alpha_prev,itrLS);
        break;
    end
    phi_alpha_prev = phi_alpha;
    alpha_prev = alpha;
    %alpha = min(alphaMax,alpha*2);
    alpha = alpha*2;
    itrLS = itrLS + 1;
end
itrLS = itrLS + itrZoom;
if itrLS >= linesearchMaxItrs
    alpha = 0;
end
end

function [alpha,itrZoom] = zoom(objFun, xk, fk, gk, pk, beta1, beta2, linesearchMaxItrs, alpha_low,alpha_high,itrLS)
phi_0 = fk;
d_phi_0 = dot(pk,gk);
itrZoom = 0;
while ( (itrLS + itrZoom) <= linesearchMaxItrs )
    itrZoom = itrZoom + 1;
    [phi_alpha_low, g_alpha_low] = objFun(xk+alpha_low*pk);
    d_phi_alpha_low = dot(pk,g_alpha_low);
    [phi_alpha_high, g_alpha_high] = objFun(xk+alpha_high*pk);
    d_phi_alpha_high = dot(pk,g_alpha_high);
    alpha = cubicInterp(alpha_low,alpha_high, phi_alpha_low, phi_alpha_high, d_phi_alpha_low, d_phi_alpha_high);
    alpha_bisect = (alpha_low + alpha_high)/2;   % want new point to be between best point lo and the bisection point
    if ~inside(alpha,alpha_low ,alpha_bisect)% it's possible that interp is infinity or even NaN: inside checks for this too.
        alpha = alpha_bisect;
    end
    [phi_alpha , g_alpha] = objFun(xk+alpha*pk);
    d_phi_alpha = dot(pk,g_alpha);
    if ( phi_alpha > phi_0 + beta1*alpha*d_phi_0 ||  phi_alpha  >= phi_alpha_low  )
        alpha_high = alpha;
    else
        if abs(d_phi_alpha) <= -beta2*d_phi_0
            return
        end
        if d_phi_alpha*(alpha_high - alpha_low) >= 0
            alpha_high = alpha_low;
        end
        alpha_low = alpha;
    end
end
end

function xmin = cubicInterp(x1, x2, f1, f2, g1, g2)
% find minimizer of the Hermite-cubic polynomial interpolating a
% function of one variable, at the two points x1 and x2, using the
% function (f1 and f2) and derivative (g1 and g2).
d1 = g1 + g2 - 3*(f1 - f2)/(x1 - x2);    % Nocedal and Wright Eqn (3.59)
d2 = sign(x2-x1)*sqrt(d1^2 - g1*g2);  % Nocedal and Wright Eqn (3.59)
xmin = x2 - (x2 - x1)*(g2 + d2 - d1)/(g2 - g1 + 2*d2);
end

function in = inside(x, a, b)
% call: in = inside(x, a, b)
% input: 3 scalars x, a, b
% output: 1 (true) if x lies between a and b
%         0 (false) if x does not lie between a and b
% Send comments/bug reports to Michael Overton, overton@cs.nyu.edu,
% with a subject header containing the string "nlcg".
% NLCG Version 1.0, 2010, see GPL license info below.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  NLCG 1.0 Copyright (C) 2010  Michael Overton
%%  This program is free software: you can redistribute it and/or modify
%%  it under the terms of the GNU General Public License as published by
%%  the Free Software Foundation, either version 3 of the License, or
%%  (at your option) any later version.
%%
%%  This program is distributed in the hope that it will be useful,
%%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%%  GNU General Public License for more details.
%%
%%  You should have received a copy of the GNU General Public License
%%  along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% this will work even if x is infinity or NaN
% but first, make sure x is real
in = 0;
if ~isreal(x)   
   return
end
if a <= b
   if x >= a && x <= b
      in = 1;
   end
else
   if x >= b && x <= a
      in = 1;
   end
end

end
% function [alpha,itrLS] = lineSearchWolfeStrong(objFun, xk, fk, gk, pk, alpha, beta1, beta2, linesearchMaxItrs)
% [f_alpha,g_alpha] = objFun(xk+alpha*pk);
% itrLS = 0;
% while( ( ( f_alpha > ( fk + alpha*beta1*gk'*pk ) ) || abs(dot(g_alpha,pk)) > beta2*abs(dot(gk,pk)) )  && itrLS < linesearchMaxItrs )
%     alpha = alpha/2;
%     [f_alpha,g_alpha] = objFun(xk+alpha*pk);
%     itrLS = itrLS + 1;
% end
% end
% 
% function [alpha,itrLS] = lineSearchWolfeWeak(objFun, xk, fk, gk, pk, alpha, beta1, beta2, linesearchMaxItrs)
% [f_alpha,g_alpha] = objFun(xk+alpha*pk);
% itrLS = 0;
% while( ( ( f_alpha > ( fk + alpha*beta1*gk'*pk ) ) || dot(g_alpha,pk) < beta2*dot(gk,pk) )  && itrLS < linesearchMaxItrs )
%     alpha = alpha/2;
%     [f_alpha,g_alpha] = objFun(xk+alpha*pk);
%     itrLS = itrLS + 1;
% end
% end