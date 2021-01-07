function h = func_1bRELAX_knownsigma(z_bar, X, Pn, Nr, Nt, Ns,t_bar)

%% Assume the path number Ns is known
%%
f1=@(x) exp(1i*2*pi*(0:Nr-1)*sin(x)/2);
f2=@(x) 1/sqrt(Nt)*exp(1i*2*pi*(0:Nt-1)*sin(x)/2);
f3=@(x) [real(x) -imag(x);imag(x) real(x)];
% f4=@(x) (f3(kron(X.'*conj(f2(x(2)).'),f1(x(1)).'))*[x(3);x(4)]-x(5)*t_bar)*sqrt(2);

% fun1=@(x) sum(-log(0.5*erfc(-z_bar.*f4(x)/sqrt(2))));

options=optimset('Display','off','tolX',1e-18);
%%
At=zeros(1,Ns);
Dt=At;
Rt=At;
It=At;
%%
N1=Nr*2;
N2=Nt*2;

W1=pi/N1/10;
W2=pi/N2/10;

% es_vec=NoSigmaNewton(interval,Nr,Nt,st,lambda,ym);
% 
% % es_vec=[A D 0.2 0.7 10]
% [s fmin1]=fmincon(fun1,es_vec,[],[],[],[],[es_vec(1)-W1 es_vec(2)-W2 -inf -inf -inf],[es_vec(1)+W1 es_vec(2)+W2 inf inf inf],[],options);
% At(1)=s(1);
% Dt(1)=s(2);
% sigma=1/s(5);
% Rt(1)=s(3)*sigma;
% It(1)=s(4)*sigma;

fmin1 = inf;
s_num=0;
sigma = sqrt(Pn);
while(s_num<Ns)
    lambdai=t_bar;
    for i=1:s_num
        lambdai=lambdai-flambda(At(i),Dt(i),Rt(i),It(i),Nr,Nt,X);
    end
    
    d=inf;
    s_num=s_num+1;
    es_n=s_num;
    flag=1;
    while(d>1e-5)
        if flag==1
%             if es_n==1
%                 es_vec=NoSigmaNewton(interval,Nr,Nt,st,lambdai,ym);
%                 f5=@(x) (f3(kron(st.'*conj(f2(x(2)).'),f1(x(1)).'))*[x(3);x(4)]-x(5)*lambdai)*sqrt(2);
%                 fun2=@(x) sum(-log(0.5*erfc(-ym.*f5(x)/sqrt(2))));
%                 [s fmin2]=fmincon(fun2,es_vec,[],[],[],[],[es_vec(1)-W1 es_vec(2)-W2 -inf -inf -inf],[At(es_n)+W1 Dt(es_n)+W2 inf inf inf],[],options);
%                 At(es_n)=s(1);
%                 Dt(es_n)=s(2);
%                 sigma=1/s(5);
%                 Rt(es_n)=s(3)*sigma;
%                 It(es_n)=s(4)*sigma;
%             else
                es_vec=SigmaNewton(Nr,Nt,X,lambdai,z_bar,sigma^2);
                f5=@(x) (f3(kron(X.'*conj(f2(x(2)).'),f1(x(1)).'))*[x(3);x(4)]-lambdai)/sigma*sqrt(2);
                fun2=@(x) sum(-log(0.5*erfc(-z_bar.*f5(x)/sqrt(2))));
                [s fmin2]=fmincon(fun2,es_vec,[],[],[],[],[es_vec(1)-W1 es_vec(2)-W2 -inf -inf],[es_vec(1)+W1 es_vec(2)+W2 inf inf],[],options);
                At(es_n)=s(1);
                Dt(es_n)=s(2);
                Rt(es_n)=s(3);
                It(es_n)=s(4);
%             end
            flag=0;
        else
%            if es_n==1
%                f5=@(x) (f3(kron(st.'*conj(f2(x(2)).'),f1(x(1)).'))*[x(3);x(4)]-x(5)*lambdai)*sqrt(2);
%                 fun2=@(x) sum(-log(0.5*erfc(-ym.*f5(x)/sqrt(2))));
%                 es_vec=[At(es_n) Dt(es_n) Rt(es_n)/sigma It(es_n)/sigma 1/sigma];
%                 [s fmin2]=fmincon(fun2,es_vec,[],[],[],[],[es_vec(1)-W1 es_vec(2)-W2 -inf -inf -inf],[At(es_n)+W1 Dt(es_n)+W2 inf inf inf],[],options);
%                 At(es_n)=s(1);
%                 Dt(es_n)=s(2);
%                 sigma=1/s(5);
%                 Rt(es_n)=s(3)*sigma;
%                 It(es_n)=s(4)*sigma;
%            else
                f5=@(x) (f3(kron(X.'*conj(f2(x(2)).'),f1(x(1)).'))*[x(3);x(4)]-lambdai)/sigma*sqrt(2);
                fun2=@(x) sum(-log(0.5*erfc(-z_bar.*f5(x)/sqrt(2))));
                es_vec=[At(es_n) Dt(es_n) Rt(es_n) It(es_n)];
                [s fmin2]=fmincon(fun2,es_vec,[],[],[],[],[At(es_n)-W1 Dt(es_n)-W2 -inf -inf],[At(es_n)+W1 Dt(es_n)+W2 inf inf],[],options);
                At(es_n)=s(1);
                Dt(es_n)=s(2);
                Rt(es_n)=s(3);
                It(es_n)=s(4);
%            end
        end
        
        %%
        if es_n==s_num
           d=abs((fmin1-fmin2)/fmin1);
           fmin1=fmin2;
        end
        es_n=mod(es_n,s_num)+1;
        lambdai=t_bar;
        for i=1:s_num
            if i~=es_n
            lambdai=lambdai-flambda(At(i),Dt(i),Rt(i),It(i),Nr,Nt,X);
            end
        end
    end
end
f_vec=@(x,y) exp(1i*pi*sin(x).*(0:y-1).');
ABS=f_vec(At,Nr);
AMS=f_vec(Dt,Nt)/sqrt(Nt);
Hv=diag(Rt+1i*It);
H=ABS*Hv*AMS';
h=H(:);
h=[real(h);imag(h)];


% Es_ML=indm(Aoa,At,Dt,Rt,It);

