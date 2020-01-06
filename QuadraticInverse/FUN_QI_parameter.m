function [A, b, mu, x_ori] = FUN_QI_parameter( M, N, density )

% A: M*N
% This function generates A, b, mu, and x_ori:
% minimize 0.5*||Ax-b||^2+mu*||x||_1

x_ori = sprandn(N,1,density); % density

sigma2 = 0e-4;

A = randn(M,N);
% A = orth(A')';
for n = 1:1:M
    A(n,:) = A(n,:) / norm(A(n,:));
end

% % ------to make the diagonal of (A'*A) equal to 1
% d_AtA = sum(A.^2, 1)'; % diagonal of AtA
% d_AtA_InvSR = d_AtA.^-0.5; % inverse square root
% % A = A * diag(d_AtA_InvSR);
% for n = 1: 1: N,
%     A(:,n) = A(:,n) * d_AtA_InvSR(n);
% end;
% % ------to make the diagonal of (A'*A) equal to 1



b = abs(abs((A * x_ori).^2 + sqrt(sigma2) * randn(M,1))); %noisy output

mu = 0.05 * max( abs( A' * b ) ); % regularization gain

A = A';