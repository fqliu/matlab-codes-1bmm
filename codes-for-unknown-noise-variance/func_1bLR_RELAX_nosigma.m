function [ht_es, BIC] = func_1bLR_RELAX_nosigma(z_bar, X, Nr, Nt, Ns, K, t_bar, N_max, aoa_true, aod_true, H_true)

% f_matrix_r=@(x) [real(x) -imag(x);imag(x) real(x)];
f_vec_r=@(x) [real(x);imag(x)];
f_steer=@(x,y) exp(1i*sin(x).'*pi.*(0:y-1).');

t = t_bar(1:K*Nr) + 1i * t_bar(K*Nr+1:2*K*Nr);
T = reshape(t,Nr,K);  %% T = unvec(t)
T_r = real(T);
T_i = imag(T);

z = z_bar(1:K*Nr) + 1i * z_bar(K*Nr+1:2*K*Nr);
Z = reshape(z,Nr,K); %% Z = unvec(z)
Z_r = real(Z);
Z_i = imag(Z);

Xt = X' / K;
Temp = T * Xt;

Denomi = sum(sum(T.*conj(T)));  %% compute the denominator in Eq. (31)
%% Initialization
sigma_hat = 1; %% initialize the noise parameter estimate
H_hat_pre = zeros(Nr,Nt); %% initialize the channel matrix estimate
% H_hat_pre = randn(Nr,Nt)+1i*randn(Nr,Nt);
epsilon1 = 1e-6;
epsilon2 = 1e-3;
relative_dif1 = 1;
gamma_tilde_hat = inf;
iter_num1 = 0;
I_max1 = 50; %% maximum iteration number
I_max2 = 10; %% maximum iteration number
while(relative_dif1 > epsilon1)
%     tic
   iter_num1 = iter_num1 + 1;
   if iter_num1 > I_max1
       break;
   end
   
   %%
   signal = H_hat_pre * X;
   %%
   S_r = Z_r .* (real(signal) - T_r) / sigma_hat;  %% Eq.(12a) (matrix version)
   S_i = Z_i .* (imag(signal) - T_i) / sigma_hat;  %% Eq.(12b)
   %%
   U_r = S_r + sqrt(2/pi) ./ erfcx(-S_r / sqrt(2));  %% Eq. (15a)
   U_i = S_i + sqrt(2/pi) ./ erfcx(-S_i / sqrt(2));  %% Eq. (15b)
   %%
   
   Q_r = U_r .* Z_r;
   Q_i = U_i .* Z_i;
   Q = Q_r + 1i * Q_i;   %% Construct Q
   
   
   
   
   Qt = Q*Xt;
   gamma_hat = 1 / sigma_hat;
   Ht1 = H_hat_pre / sigma_hat;
   
   iter_num2 = 0;
   relative_dif2 = 1;
   while(relative_dif2 > epsilon2)
       
       iter_num2 = iter_num2 + 1;
       if iter_num2 > I_max2
           break;
       end
       %% SVT algorithm
       
       Ht2 = Qt + Temp * gamma_hat;
       [S V D] = svd(Ht2);
       N_min = min(Nr, Nt);
       v = diag(V); 
       mu = sqrt(2*Nr) / sqrt(K) / gamma_tilde_hat;  %% Eq. (49)
       v = v - mu;
%        vt=vt-sqrt(2*Nr)/sqrt(K)*sigmat;
       v(find(v < 0)) = 0;
       Ht2 = S(:,1:N_min) * diag(v) * D(:,1:N_min)';    %% Eq. (37)
       
       %%
          
       A = Ht2 * X;
       Y_tilde = Q + T * gamma_hat;
       N_tilde = Y_tilde - A;
       
       gamma_tilde_hat = sqrt(2*K*Nr / sum(sum(N_tilde.*conj(N_tilde))));    %% Eq. (46)
       %%
       B = Q - A;
       gamma_hat = - real( sum ( sum( B .* conj(T) ) ) ) / Denomi;  %% Eq. (31)
       sigma_hat = 1 / gamma_hat;
        
       %%
       Tem_d = (Ht1 - Ht2) / gamma_hat;
       relative_dif2 = sum(sum(Tem_d.*conj(Tem_d))) / Nr;
       Ht1 = Ht2;
   end
   
   H_hat = Ht2 / gamma_hat;
   Tem_d = H_hat_pre - H_hat;
   relative_dif1 = sum(sum(Tem_d.*conj(Tem_d))) / Nr;
   H_hat_pre = H_hat;

end


%% 
N_theta = Nr*2;
N_varphi = Nt*2;

ht = fft2(H_hat, N_theta, N_varphi);

aoat = [(1:N_theta/2)-1 -N_theta/2:-1]/N_theta;
aodt = [(1:N_varphi/2)-1 -N_varphi/2:-1]/N_varphi;
aoat = asin(aoat*2);
aodt = -asin(aodt*2);

aoat2 = aoat/pi*180;
aodt2 = aodt/pi*180;

ht_abs = abs(ht).^2/Nr^2/Nt;

Ht_temp = zeros(N_theta+2,N_varphi+2);
Ht_temp(2:N_theta+1,2:N_varphi+1) = ht_abs;
peaks_v = zeros(N_theta*N_varphi, 1);
peaks_x = zeros(N_theta*N_varphi, 1);
peaks_y = zeros(N_theta*N_varphi, 1);
peaks_l = 0; %% number of peaks

%% Find Peaks
for i = 1 : N_theta
    for j=1:N_varphi
        ii = i + 1;
        jj = j + 1;
        if (Ht_temp(ii,jj)>Ht_temp(ii-1,jj)&&Ht_temp(ii,jj)>Ht_temp(ii,jj-1)&&Ht_temp(ii,jj)>Ht_temp(ii+1,jj)&&Ht_temp(ii,jj)>Ht_temp(ii,jj+1))
            peaks_l = peaks_l + 1;
            peaks_v(peaks_l) = Ht_temp(ii,jj);
            peaks_x(peaks_l) = i;
            peaks_y(peaks_l) = j;
        end
    end
end
[~,ind]=sort(peaks_v, 'descend');
peaks_x = peaks_x(ind);
peaks_y = peaks_y(ind);
aoa_x = aoat(peaks_x(1:peaks_l));  
aod_y = aodt(peaks_y(1:peaks_l));



if peaks_l>0
%% 1bBIC and 1bRELAX
f_BIC=@(x,y) 2 * x + 6*y*log(Nr*Nt) + 4*y*log(K);
% N_max = 8;  %% maximum possible path number
BIC = zeros(N_max,1);
aoa_es=zeros(peaks_l,1);
aod_es=aoa_es;
alpha_r_es=aoa_es;
alpha_i_es=aoa_es;


path_number = 1;
% tic
[aoa_refine aod_refine alpha_r_refine alpha_i_refine sigma_v loglihood]=f_nosigma_newton_refine(aoa_x(path_number),aod_y(path_number),Nr,Nt,K,z_bar,X,t_bar);
aoa_es(path_number)=aoa_refine;
aod_es(path_number)=aod_refine;
alpha_r_es(path_number)=alpha_r_refine;
alpha_i_es(path_number)=alpha_i_refine;
% toc

aoa_hat_real=aoa_es(1:path_number);
aod_hat_real=aod_es(1:path_number);
alpha_hat_real = alpha_r_es(1:path_number)+1i*alpha_i_es(1:path_number);



BIC(path_number)=f_BIC(loglihood,path_number);

while(path_number < N_max)
    %% Thresholds
    ABS_t=f_steer(aoa_es(1:path_number),Nr);
    AMS_t=1/sqrt(Nt)*f_steer(aod_es(1:path_number),Nt);
    Hv = diag(alpha_r_es(1:path_number)+ 1i*alpha_i_es(1:path_number));
    Yt=ABS_t* Hv * AMS_t' *X;
    yt = Yt(:);
    lambdat = t_bar - f_vec_r(yt);
%     lambdat=lambda;
    %%
    path_number = path_number+1;
    es_num = path_number;
%     tic
    [aoa_refine aod_refine alpha_r_refine alpha_i_refine loglihood]=f_sigma_newton_refine(aoa_x(path_number),aod_y(path_number),Nr,Nt,K,z_bar,X,lambdat,sigma_v);
%     toc
    aoa_es(es_num)=aoa_refine;
    aod_es(es_num)=aod_refine;
    alpha_r_es(es_num)=alpha_r_refine;
    alpha_i_es(es_num)=alpha_i_refine;
    log_v1=loglihood;
    %% Iteritive Refine
    dif = inf;
    while(dif > 1e-4)
        es_num = mod(es_num,path_number)+1;
        ABS_t = f_steer(aoa_es(1:path_number),Nr);
        AMS_t = 1/sqrt(Nt)*f_steer(aod_es(1:path_number),Nt);
        Hv = diag(alpha_r_es(1:path_number)+1i*alpha_i_es(1:path_number));
        Yt = ABS_t*Hv*AMS_t'*X;
        Yt = Yt-1/sqrt(Nt)*(alpha_r_es(es_num)+1i*alpha_i_es(es_num))*f_steer(aoa_es(es_num),Nr)*f_steer(aod_es(es_num),Nt)'*X;
        yt = Yt(:);
        lambdat = t_bar-f_vec_r(yt);
%         lambdat=lambda;
        if es_num==1
            [aoa_refine aod_refine alpha_r_refine alpha_i_refine sigma_v loglihood] = f_nosigma_newton_refine(aoa_es(es_num),aod_es(es_num),Nr,Nt,K,z_bar,X,lambdat);
            aoa_es(es_num)=aoa_refine;
            aod_es(es_num)=aod_refine;
            alpha_r_es(es_num)=alpha_r_refine;
            alpha_i_es(es_num)=alpha_i_refine;
        else
            [aoa_refine aod_refine alpha_r_refine alpha_i_refine loglihood] = f_sigma_newton_refine(aoa_es(es_num),aod_es(es_num),Nr,Nt,K,z_bar,X,lambdat,sigma_v);
            aoa_es(es_num)=aoa_refine;
            aod_es(es_num)=aod_refine;
            alpha_r_es(es_num)=alpha_r_refine;
            alpha_i_es(es_num)=alpha_i_refine;
        end
        
        if (es_num==path_number)
            log_v2 = loglihood;
            dif = abs(log_v1-log_v2)/abs(log_v1);
            log_v1 = log_v2;
        end
    end
    
    BIC(path_number)=f_BIC(loglihood,path_number);
    if BIC(path_number)<BIC(path_number-1)
        aoa_hat_real=aoa_es(1:path_number);
        aod_hat_real=aod_es(1:path_number);
        alpha_hat_real=alpha_r_es(1:path_number)+1i*alpha_i_es(1:path_number);
%         break;
    end

end

%% Plot 3D Fig.
figure(1);
[X, Y]=meshgrid(aoat2,aodt2);
mesh(X,Y,ht_abs.',2*ones(size(ht_abs.')),'EdgeColor','b');
hold on;
for i=1:Ns
    p1=plot3(aoa_true(i),aod_true(i),abs(H_true(i)).^2,'ro','MarkerSize',8,'LineWidth',1);
    plot3(aoa_true(i)*ones(1,100),aod_true(i)*ones(1,100),0:abs(H_true(i)).^2/99:abs(H_true(i))^2,'--r');
end
for i=1:Ns
    p2=plot3(aoa_hat_real(i)/pi*180,aod_hat_real(i)/pi*180,abs(alpha_hat_real(i)).^2,'kx','MarkerSize',8,'LineWidth',1);
    plot3(aoa_hat_real(i)*ones(1,100)/pi*180,aod_hat_real(i)*ones(1,100)/pi*180,0:abs(alpha_hat_real(i)).^2/99:abs(alpha_hat_real(i))^2,'--k');
end
xlabel('AoA (Deg)');
ylabel('AoD (Deg)');
zlabel('Spatial Power');
legend([p1 p2],'True Value','1bLR-RELAX');


% path_number=path_number-1;
ABS_t = f_steer(aoa_hat_real,Nr);
AMS_t = 1/sqrt(Nt)*f_steer(aod_hat_real,Nt);
Hv = diag(alpha_hat_real);
Ht = ABS_t*Hv*AMS_t';
h = Ht(:);
ht_es = f_vec_r(h);
% path_BIC=0;
% if length(aoa_real)==Ns
%     path_BIC=1;
% end
end

