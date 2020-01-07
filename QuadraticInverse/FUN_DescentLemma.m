function [sol] = FUN_DescentLemma(A, y, mu, K, N, x0, algsetup, discount)

% Implementation of the Bregman proximal gradient descent algorithm (BPGD) proposed in
% First order methods beyond convexity and Lipschitz gradient continuity with applications
% to quadratic inverse problems, by Bolte, Sabach, Teboulle and Vaisbourd

algType       = algsetup.algorithm;
MaxIter_inner = algsetup.MaxIter_inner;
stepsize      = algsetup.stepsize;
MaxIter_outer = algsetup.MaxIter_outer;

objval  = zeros(1, MaxIter_outer + 1);
cputime = zeros(1, MaxIter_outer + 1);

x   = x0;

% initialization
tic;

Anorm = sum(A.^2, 1);
L     = 3 * sum(Anorm.^2) + Anorm * abs(y);

% for n = 1: 1: N
%     An = A(:, n);
%     normAn = An' * An;
%     L = L + 3 * normAn^2 + normAn * abs(y(n));
% end

L      = discount * L;
lambda = 1 / L;

ATx  = A' * x;
loss = ATx.^2 - y;

cputime(1) = toc;

objval(1)  = 0.25*sum(loss.^2) + mu * sum(abs(x));

disp(algsetup.legend);
disp(['iteration ' num2str(0) ', value ' num2str(objval(1))]);

for t = 1: 1: MaxIter_outer
    tic;
    
    gradient_g = A * (ATx .* loss);
    gradient_h = (x' * x) * x + x;
    p          = lambda * gradient_g - gradient_h;
    vx         = max(p - lambda * mu * ones(K,1), zeros(K,1)) - max(-p - lambda * mu * ones(K,1), zeros(K, 1));

    t_star     = root_polynomial3(vx'*vx);

    x          = -t_star * vx;

    ATx  = A' * x;
    loss = ATx.^2 - y;
    
    cputime(t+1) = cputime(t) + toc;

    objval(t+1) = 0.25 * sum(loss.^2) + mu * sum(abs(x));

%     disp(['iteration ' num2str(t+1) ', value ' num2str(objval(t+1)) ', and judge ' num2str(objval(t+1)<=objval(t))]);
%     if objval(t+1) > objval(t)
%         disp(['root value ' num2str(t_star) ', accuracy ' num2str(value_polynomial3(vx'*vx, t_star))]);
%     end
end

sol.objval  = objval;
sol.x       = x;
sol.cputime = cputime;

end

function gamma = root_polynomial3(c4)
    % 1/4 * c4 * t^4 + 1/2 * t^2 - t
    Sigma1 = 1/2/c4;
    Sigma2 = 1/3/c4;
    Sigma3 = Sigma1^2 + Sigma2^3;
    Sigma3_sqrt = sqrt(Sigma3);
    gamma = nthroot(Sigma1 + Sigma3_sqrt,3) + nthroot(Sigma1 - Sigma3_sqrt,3);   

%     % performing the exact line search the stepsize by Matlab
%     f       = @(x) c4/4 * x^4 + 1/2 * x^2 - x;
%     options = optimoptions('fmincon','Display','off','Algorithm','sqp');
%     gamma   = fmincon(f,0,[],[],[],[],0,[],[],options);

    gamma = max(gamma, 0); % for numerical stability
end

function funval = value_polynomial3(c4, t)
    funval = t^3 * c4 + t - 1;
end