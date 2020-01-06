function [sol] = FUN_blockQI(A, y, mu, K, x0, algsetup)

algType       = algsetup.algorithm;
MaxIter_inner = algsetup.MaxIter_inner;
stepsize      = algsetup.stepsize;
MaxIter_outer = algsetup.MaxIter_outer;

objval  = zeros(1, MaxIter_outer + 1);
cputime = zeros(1, MaxIter_outer + 1);

P   = algsetup.P;
Kp  = K / P;
x   = x0;
tau = 0.0001;

% initialization
tic;
ATx  = A' * x;
loss = ATx.^2 - y;

cputime(1) = toc;
objval(1)  = 0.25*sum(loss.^2) + mu * sum(abs(x));

disp(algsetup.legend);

for t = 1: 1: MaxIter_outer
    tic;
    for k = 1: 1: P
        xk = x((k - 1) * Kp + 1: k * Kp);
        Ak = A((k - 1) * Kp + 1: k * Kp, :); % Kp*N

        if strcmp(algType, 'sca')
            bk = tau * xk + 2 * Ak * ((ATx.^2).*(Ak'*xk)) - Ak * (ATx .* loss);
            xk_hat = FUN_lasso_sca(Ak, ATx.^2, tau, bk, mu, xk, Kp, MaxIter_inner);
        elseif strcmp(algType, 'gra') 
            bk = tau * xk - Ak * (ATx .* loss);
            xk_hat = FUN_lasso_gra(tau, bk, mu, xk, Kp, 1);
        end

        xk_dif   = (xk_hat - xk);
        AkTx_dif = Ak' * xk_dif;
        
        if strcmp(stepsize, 'exact')
            gamma = exactLineSearch(y, ATx, AkTx_dif, mu, xk_hat, xk);
        elseif strcmp(stepsize, 'constant')
            gamma = 0.1;
        elseif strcmp(stepsize, 'successive')
            descent = (Ak * (ATx .* loss))' * (xk_hat - xk) + mu*(norm(xk_hat,1) - norm(xk,1));
            gamma = armijo(y, mu, loss, xk, xk_hat, ATx, AkTx_dif, descent);
        end
        
        x((k - 1) * Kp + 1: k * Kp) = xk + gamma * xk_dif;
        ATx  = ATx + gamma * AkTx_dif;
        loss = ATx.^2 - y;
        
        objval(t+1) = 0.25*sum(loss.^2) + mu * sum(abs(x));
    end    
    cputime(t+1) = cputime(t) + toc;
end

sol.objval  = objval;
sol.cputime = cputime;
sol.x       = x;

function gamma = armijo(y, mu, loss, xk, xk_hat, ATx, AkTx_dif, descent)
alpha   = 0.01;
beta    = 0.5;
gamma   = 1;
gxt     = mu * sum(abs(xk));
gxt_hat = mu * sum(abs(xk_hat));
f       = sum(loss.^2);
while(1)
    ATx_tmp    = ATx + gamma * AkTx_dif;
    loss_tmp   = ATx_tmp.^2 - y;
    f_tmp = sum(loss_tmp.^2) + gamma * (gxt_hat - gxt);
%     disp([ num2str(f_tmp) ', ' num2str(f) ', ' num2str(f + alpha * gamma * descent)]);
    if (f_tmp <= f + alpha * gamma * descent)
        break
    else
        gamma = gamma * beta;
    end
end

function gamma = exactLineSearch(y, ATx, AkTx_dif, mu, xk_hat, xk)
c4 = sum(AkTx_dif.^4);
% disp(['iteration ' num2str(t) ', descent ' num2str(descent) ' and variable difference ' num2str(c4)]);
if c4<=10^-5
    gamma = 1;
else
    c3     = 3 * sum((AkTx_dif.^3) .* ATx);
    c2     = sum((AkTx_dif.^2) .* (3 * ATx.^2 -  y));
    c1     = sum(AkTx_dif .* ATx .* (ATx.^2 - y)) + mu * (norm(xk_hat,1) - norm(xk,1));
    Sigma1 = (-(c3/3/c4)^3 + c3*c2/6/c4^2 - c1/2/c4);
    Sigma2 = c2/3/c4 - (c3/3/c4)^2;
    Sigma3 = Sigma1^2 + Sigma2^3;
    Sigma3_sqrt = sqrt(Sigma3);
    if Sigma3 >= 0
        gamma = nthroot(Sigma1 + Sigma3_sqrt,3)...
            + nthroot(Sigma1 - Sigma3_sqrt,3)...
            - c3/3/c4;
    else
        C1 = 1; C1(4) = -(Sigma1 + Sigma3_sqrt);
        C2 = 1; C2(4) = -(Sigma1 - Sigma3_sqrt);
        R = real(roots(C1) + roots(C2)) - c3/3/c4 * ones(3,1);
        gamma = min(R(R>0));
        clear C1 C2 R;
    end
end
gamma = max(0, min(gamma, 1));