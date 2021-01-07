function h = func_unqt_LR_knownsigma(Y, X, Nr, Nt, K, Pn)

sigma = sqrt(Pn/2);
gamma = 1 / sigma;

Xt = X' / (X*X');
Temp = Y * Xt;
% N = Y - Temp*X;
% gamma_hat = sqrt(2*K*Nr/sum(sum(N.*conj(N))));

[S V D] = svd(Temp);
v = diag(V); 

mu = sqrt(2*Nr)/sqrt(K)/gamma;
v = v - mu;
v(find(v < 0)) = 0;
N_min = min(Nr, Nt);
H_hat = S(:,1:N_min) * diag(v) * D(:,1:N_min)';

%%
h = reshape(H_hat , Nr*Nt, 1);
h = [real(h); imag(h)];

