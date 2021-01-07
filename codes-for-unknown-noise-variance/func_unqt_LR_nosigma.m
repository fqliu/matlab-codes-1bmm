function h = func_unqt_LR_nosigma(Y, X, Nr, Nt, K)



Xt = X' / (X*X');
Temp = Y * Xt;
N = Y - Temp*X;
gamma_hat = sqrt(2*K*Nr / sum(sum(N.*conj(N))));

[S V D] = svd(Temp);
vt = diag(V); 


H_hat_pre = zeros(Nr,Nt);

dif = 1;
epsilon = 1e-6;
iter_num = 0;
I_max = 50; %% maximum iteration number
%%
while(dif > epsilon)
    iter_num = iter_num + 1;
    if iter_num > I_max
       break;
    end
    
%     mu = sqrt(2*Nr)/sqrt(K)*gamma_hat;
%     v = vt - mu/gamma_hat^2;
    mu = sqrt(2*Nr)/sqrt(K)/gamma_hat;
    v = vt - mu;
    v(find(v < 0))=0;
    N_min = min(Nr, Nt);
    H_hat = S(:,1:N_min) * diag(v) * D(:,1:N_min)';
    N = Y - H_hat * X;
    gamma_hat = sqrt(2*K*Nr/sum(sum(N.*conj(N))));
    
    Tem_d = H_hat_pre - H_hat;
    dif = sum(sum(Tem_d.*conj(Tem_d)))/Nr;
    H_hat_pre = H_hat;
end
%%
h = reshape(H_hat , Nr*Nt, 1);
h = [real(h); imag(h)];

