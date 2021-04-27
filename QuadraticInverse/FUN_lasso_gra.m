function [x, output_success, output_val, output_error] = FUN_lasso_gra(tau, b, mu, x, K, MaxIter)

% This function solves the following optimization problem:
% min_x 0.5 * tau * x' * x - b' * x + mu * ||x||_1
% Input variables:
%   tau: a positive constant
%   x: initial guess of the optimal point (starting point of the iterative algorithm)
%   K: dimension of x
%   MaxIter: maximum number of iterations

if nargin < 6
    MaxIter = 10000;
end


% set up
output_error=zeros(1,MaxIter);
output_success=0;

mu_vec_normalized = mu / tau * ones(K,1);

Gradient = tau * x - b;

% formal algorithm
for t = 1: 1: MaxIter   
    % compute best-response    
    x_hat = FUN_quad(x - Gradient / tau, mu_vec_normalized, K);
    
    % compute stepsize
    x_dif  = x_hat-x;
    Ax_dif = tau * x_dif;

    stepsize = max(0, min((-Gradient' * x_dif - mu * (sum((abs(x_hat) - abs(x)))))/(x_dif' * Ax_dif), 1)); % sum(abs(x)) is faster than norm(x,1)

    % update variable
    x = x + stepsize * x_dif;
    Gradient = Gradient + stepsize * Ax_dif;
    
    % calculate variable precision
    output_error(t) = sqrt(x_dif' * x_dif); % error in variable
    
    %     calculate objective value
    output_val = 1/2 * x' * (Gradient - b) + mu * sum(abs(x));
        
%     disp(['iteration ' num2str(t) ' with value ' num2str(output_val)...
%         ' and stepsize ' num2str(stepsize) ' and precision ' num2str(output_error(t))]);

    % check if stopping criterion is satisfied or not
    if output_error(t)<10^-6
        output_success=1;
        break; 
    end
    
end

% cvx_begin
%     variable xc(K)
%     minimize(0.5 * xc' * A * xc - b' * xc + mu*norm(xc,1))
% cvx_end
% disp(['STELA ' num2str(output_val) ', CVX ', num2str(cvx_optval)]);

function [x]=FUN_quad(q, t, K)

% min_x 0.5* x^2 - q * x + t * |x| (t>0)

x = max(q - t, zeros(K,1)) - max(-q - t, zeros(K,1));