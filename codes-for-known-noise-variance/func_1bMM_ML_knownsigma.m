function h = func_1bMM_ML_knownsigma(z_bar, X, Nr, Nt, K, t_bar, Pn)

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


Xt = X'/(X*X');  %% Eq. (20)

Temp = T * Xt;

%% Initialization
H_hat_pre = zeros(Nr,Nt); %% initialize the channel matrix estimate
% H_hat_pre =randn(Nr,Nt)+1i*randn(Nr,Nt);
epsilon = 1e-6;
relative_dif = 1;

iter_num = 0;
I_max = 50; %% maximum iteration number

while(relative_dif > epsilon)
%     tic
   iter_num = iter_num + 1;
   if iter_num > I_max
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
   %%
   H_hat = Q * Xt / gamma + Temp;  %% Eq. (19) /gamma

   
   %%
   Dif = H_hat_pre- H_hat;
   
   relative_dif = sum(sum( Dif.*conj(Dif))) / Nr;

   H_hat_pre = H_hat;
   
end

h = reshape(H_hat, Nr*Nt, 1);
h = [real(h);imag(h)];



