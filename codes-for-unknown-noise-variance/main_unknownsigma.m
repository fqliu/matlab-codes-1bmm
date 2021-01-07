clear all;
close all;
clc;

%% System parameters
Nr = 64;                     %% receive antenna number
Nt = 32;                     %% transmit antenna number
K = 128;                     %% pilot length
SNR_dB = 10;                 %% SNR
SNR = 10^(SNR_dB/10);

Pn = 1 / SNR;               %% noise power (the transmit power is normalized as 1)
Ns = 4;                     %% scattering path number
N_max = 8;                  %% maximum possible path number (used in 1bLR-RELAX)

%% AoA and AoD

% aoa = [47 44 -29 -71];
% aod = [-57 -55 57 18];
aoa = [47 20 -29 -71];
aod = [-57 -34 57 18];
aoa = aoa / 180 * pi;
aod = aod / 180 * pi;

%% Thresholds vector (real value) 
 
h_max = sqrt(1 + Pn); 
% Note that in practice the average received signal power Py = 1 + Pn 
% can be easily measured from the radio frequency (RF) circuit at the
% antenna output before quantization, e.g., using the automatic gain control (AGC) circuit
ht = - h_max : 2 * h_max / 7 : h_max;
ht = ht.';

t_bar = zeros(2*K*Nr, 1);
kro = ones(K, 1);
tem = ht(randi(8, Nr, 1));
t_bar(1 : K*Nr) = kron(kro, tem); % real part
tem = ht(randi(8, Nr, 1));
t_bar(K * Nr + 1 : 2 * K * Nr) = kron(kro, tem);  % imaginary part

%% Angular-domain channel model

alpha = [1-1*1i -0.9+0.9*1i 1-1*1i 0.8+0.8*1i]; %% path gains
alpha = alpha / norm(alpha);
Hv = diag(alpha);

f_vec=@(x,y) exp(1i*pi.*sin(x).*(0:y-1).');

ABS = f_vec(aoa,Nr);           %% receive steering matrix
AMS = f_vec(aod,Nt)/sqrt(Nt);  %% transmit steering matrix
H = ABS * Hv * AMS';
h = H(:);
h = [real(h); imag(h)];
H_fro = norm(H,'fro')^2;

%% ZadoffZhu sequence / Orthogonal Pilots

x = zadoff_chu(K);
S0 = zeros(K,K);
ind = 1:K;
for i = 1:K
    ind = mod(ind,K) + 1;
    S0(:,i) = x(ind);
end
X = S0(randperm(Nt),:);

%% Random QPSK pilots

% qpsk_signal = randi(4,Nt,K);
% X = pskmod(qpsk_signal - 1, 4, pi/4);

%% One-bit quantization

N = random('norm', 0, sqrt(Pn/2), Nr, K) + 1i * random('norm', 0, sqrt(Pn/2), Nr, K); %% Noise matrix
Y = H * X + N;
y = Y(:);
y_bar = [real(y); imag(y)];
z_bar = sign(y_bar - t_bar);

%% Channel Estimation 

%% One-bit channel estimation
% Maximum likelihood channel estimation
h_1bMM_ML = func_1bMM_ML_nosigma(z_bar, X, Nr, Nt, K, t_bar);          %% 1bMM-ML

% Low-rank channel estimation
h_1bMM_LR = func_1bMM_LR_nosigma(z_bar, X, Nr, Nt, K, t_bar);          %% 1bMM-LR with orthogonal pilots
% h_1bMM_LR = func_1bMM_LR_nosigma2(z_bar, X, Nr, Nt, K, t_bar);       %% 1bMM-LR with non-orthogonal pilots

% Angular-domain channel estimation
[h_1bLR_RELAX, BIC_value] = func_1bLR_RELAX_nosigma(z_bar, X, Nr, Nt, Ns, K, t_bar, N_max, aoa/pi*180, aod/pi*180, diag(Hv));        %% 1bLR-RELAX with orthogonal pilots
% [h_1bLR_RELAX, BIC_value] = func_1bLR_RELAX_nosigma2(z_bar, X, Nr, Nt, Ns, K, t_bar, N_max, aoa/pi*180, aod/pi*180, diag(Hv));     %% 1bLR-RELAX with non-orthogonal pilots

h_1bRELAX = func_1bRELAX_nosigma(z_bar, X, Nr, Nt, Ns, t_bar);
 

%% Unquantized channel estimation
% Maximum likelihood channel estimation
h_unqt_ML  = func_unqt_ML(Y, X, Nr, Nt);

% Low-rank channel estimation
h_unqt_LR = func_unqt_LR_nosigma(Y, X, Nr, Nt, K);  %% UnqtLR with orthogonal pilots
% h_unqt_LR = func_unqt_LR_nosigma2(Y, X, Nr, Nt, K);  %% UnqtLR with non-orthogonal pilots

%% Mean-squared error
sum(abs(h-h_1bMM_ML).^2)
sum(abs(h-h_1bMM_LR).^2)
sum(abs(h-h_1bLR_RELAX).^2)
sum(abs(h-h_1bRELAX).^2)
sum(abs(h-h_unqt_ML).^2)
sum(abs(h-h_unqt_LR).^2)


%% Bayesian information criterion for path number detection
figure(2);
plot( 1 : N_max, BIC_value, '-ko', 'LineWidth', 1.5, 'MarkerSize', 8);
h1 = xlabel('Assumed Path Number n');
ylabel('1bBIC Cost Function Value of (57)');
set(gca, 'FontSize', 12);
grid on;

