function h = func_1bMM_LR_knownsigma2(z_bar, X, Nr, Nt, K, t_bar, Pn)


sigma = sqrt(Pn/2);
gamma = 1 / sigma;

t = t_bar(1:K*Nr) + 1i * t_bar(K*Nr+1:2*K*Nr);
T = reshape(t,Nr,K);  %% T = unvec(t)
T_r = real(T);
T_i = imag(T);

z = z_bar(1:K*Nr) + 1i * z_bar(K*Nr+1:2*K*Nr);
Z = reshape(z,Nr,K); %% Z = unvec(z)
Z_r = real(Z);
Z_i = imag(Z);


Xt = X * X';
Temp =T * X';

%% Compute tau in Eq. (38)
[E V D] = eig(conj(Xt));
v = diag(V);
tau = 1 / (2*max(v));

%% Initialization
H_hat_pre = zeros(Nr,Nt); %% initialize channel matrix estimate
% H_hat_pre = randn(Nr,Nt)+1i*randn(Nr,Nt);
epsilon1 = 1e-6;
epsilon2 = 1e-3;
epsilon3 = 1e-3;
relative_dif1 = 1;
gamma_tilde = inf;
iter_num1 = 0;
I_max1 = 50; %% maximum iteration number
I_max2 = 10; %% maximum iteration number
I_max3 = 10; %% maximum iteration number
% fmin1=inf;

while(relative_dif1 > epsilon1)
%     tic
   iter_num1 = iter_num1 + 1;
   if iter_num1 > I_max1
       break;
   end
   
   %%
   signal = H_hat_pre * X;
   %%
   S_r = Z_r .* (real(signal) - T_r) / sigma;  %% Eq.(12a) (matrix version)
   S_i = Z_i .* (imag(signal) - T_i) / sigma;  %% Eq.(12b)
   %%
   U_r = S_r + sqrt(2/pi) ./ erfcx(-S_r / sqrt(2));  %% Eq. (15a)
   U_i = S_i + sqrt(2/pi) ./ erfcx(-S_i / sqrt(2));  %% Eq. (15b)
   %%
   
   Q_r = U_r .* Z_r;
   Q_i = U_i .* Z_i;
   Q = Q_r + 1i * Q_i;   %% Construct Q

   
   Qt = Q*X';
   Ht_pre = H_hat_pre * gamma;
   
   
   Ht_ex = H_hat_pre * gamma;
   
   Ht2 = H_hat_pre;
   
   iter_num2 = 0;
   relative_dif2 = 1;
   while(relative_dif2 > epsilon2)
       
       iter_num2 = iter_num2 + 1;
       if iter_num2 > I_max2
           break;
       end
       
       iter_num3 = 0;
       relative_dif3 = 1;
       
       t1 = 1;
       while(relative_dif3 > epsilon3)
           
           iter_num3 = iter_num3 + 1;
           if iter_num3 > I_max3
              break;
           end
           
           %% SVT algorithm
           R = Ht_ex - tau * Ht_ex * Xt + tau * (Temp * gamma + Qt); %% Eq. (40)
           [S, V, D] = svd(R);
           
           N_min = min(Nr, Nt);
           v = diag(V); 
           mu = sqrt(2*Nr) / sqrt(K) / gamma_tilde;  %% Eq. (49)
           v=v - K * tau * mu;    
           v(find(v < 0)) = 0;
           Ht_hat = S(:,1:N_min) * diag(v) * D(:,1:N_min)';  %% Eq. (41)
           %%
           
           Tem_d = (Ht_pre - Ht_hat) / gamma;
           
           relative_dif3 = sum(sum(Tem_d .* conj(Tem_d))) / Nr;

           t2 = (1+ sqrt(1 + 4 * t1^2)) / 2;
           Ht_ex = Ht_hat + (t1 - 1) / t2 * (Ht_hat - Ht_pre);
           
           Ht_pre = Ht_hat;
           t1 = t2;
           
       end
%        iter_num3
       A = Ht_hat*X;
              %%
       Y_tilde = Q + T * gamma;
       N_tilde = Y_tilde - A;
       gamma_tilde = sqrt(2*K*Nr/sum(sum(N_tilde.*conj(N_tilde))));    %% Eq. (46)
             
       Tem_d = Ht2 - Ht_hat / gamma;
       relative_dif2 = sum(sum(Tem_d .* conj(Tem_d))) / Nr;
       Ht2 = Ht_hat / gamma;
   end

   
   Ht_hat = Ht_hat / gamma;
   Tem_d = H_hat_pre - Ht_hat;
   relative_dif1 = sum(sum(Tem_d .* conj(Tem_d))) / Nr;
   H_hat_pre = Ht_hat;

end

h=reshape(H_hat_pre, Nr*Nt, 1);
h=[real(h); imag(h)];

