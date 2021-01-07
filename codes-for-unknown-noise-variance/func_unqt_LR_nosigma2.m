function h = func_unqt_LR_nosigma2(Y, X, Nr, Nt, K)

Xt = X * X';
[E V D] = eig(conj(Xt));
v = diag(V);
tau = 1/(2*max(v));

gamma_hat = inf; 
 
H_hat_pre2 = zeros(Nr,Nt);

% tem=1.4;
%%
dif = 1;
epsilon1 = 1e-6;
iter_num1 = 0;
I_max1 = 50; %% maximum iteration number

epsilon2 = 1e-3;

I_max2 = 10; %% maximum iteration number
while(dif > epsilon1)
    iter_num1 = iter_num1 + 1;
    
    if iter_num1 > I_max1
       break;
    end
    
    t1 = 1;
    
    dif2 = 1;
    
    iter_num2 = 0;
    
    Ht_ex = H_hat_pre2;
    H_hat_pre = H_hat_pre2;
    
    while(dif2 > epsilon2)
        iter_num2 = iter_num2 + 1;
        
        if iter_num2 > I_max2
            break;
        end
        
        R = Ht_ex - tau * Ht_ex * Xt + tau * Y * X';
        
        mu = sqrt(2*Nr) / sqrt(K) / gamma_hat;
        %     mu = K * sqrt(2*Nr)/sqrt(K) / gamma_hat * tau;
        
        [S, V, D] = svd(R);
        
        v = diag(V(1:Nt,1:Nt));
        
        v = v - K * tau * mu;
        
        v(find( v < 0)) = 0;
        
        
        N_min = min(Nr, Nt);
        
        H_hat = S(:,1:N_min) * diag(v) * D(:,1:N_min)';  
        
        Tem_d = H_hat_pre - H_hat;
        
        dif2 = sum(sum( Tem_d .* conj(Tem_d))) / Nr;
        
        t2 = (1 + sqrt(1 + 4 * t1^2)) / 2;
        
        Ht_ex = H_hat + (t1 - 1) / t2 * (H_hat - H_hat_pre);
        
        t1 = t2;
        
        H_hat_pre = H_hat;
        
    end
%     iter_num2 
    N = Y - H_hat * X;
    gamma_hat = sqrt( 2*K*Nr / sum(sum(N .* conj(N))));  %% Eq. (43)
    
    Tem_d = H_hat_pre2 - H_hat;
    dif = sum(sum(Tem_d .* conj(Tem_d))) / Nr;

    H_hat_pre2 = H_hat;
end
%%
h = reshape(H_hat, Nr*Nt, 1);
h = [real(h); imag(h)];

