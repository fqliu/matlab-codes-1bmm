function h = func_unqt_LR_knownsigma2(Y, X, Nr, Nt, K, Pn)

sigma = sqrt(Pn/2);
gamma = 1 / sigma;

Xt = X * X';
[E V D] = eig(conj(Xt));
v = diag(V);
tau = 1/(2*max(v));

H_hat_pre = zeros(Nr,Nt);

%%

epsilon = 1e-6;
I_max = 50; %% maximum iteration number
iter_num = 0;
Ht_ex = H_hat_pre;
dif = 1;
t1 = 1;
while(dif > epsilon)
    iter_num = iter_num + 1;
    
    if iter_num > I_max
        break;
    end
    
    R = Ht_ex - tau * Ht_ex * Xt + tau * Y * X';
    
    mu = sqrt(2*Nr) / sqrt(K) / gamma;
    %     mu = K * sqrt(2*Nr)/sqrt(K) / gamma_hat * tau;
    
    [S, V, D] = svd(R);
    
    v = diag(V(1:Nt,1:Nt));
    
    v = v - K * tau * mu;
    
    v(find( v < 0)) = 0;
    
    
    N_min = min(Nr, Nt);
    
    H_hat = S(:,1:N_min) * diag(v) * D(:,1:N_min)';
    
    Tem_d = H_hat_pre - H_hat;
    
    dif = sum(sum( Tem_d .* conj(Tem_d))) / Nr;
    
    t2 = (1 + sqrt(1 + 4 * t1^2)) / 2;
    
    Ht_ex = H_hat + (t1 - 1) / t2 * (H_hat - H_hat_pre);
    
    t1 = t2;
    
    H_hat_pre = H_hat;
    
end
%%
h = reshape(H_hat, Nr*Nt, 1);
h = [real(h); imag(h)];

