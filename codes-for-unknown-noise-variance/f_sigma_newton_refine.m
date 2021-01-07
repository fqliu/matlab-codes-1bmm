function [aoa aod alpha_r alpha_i loglihood]=f_sigma_newton_refine(aoat,aodt,Nr,Nt,K,z,X,lambda,sigma)
f11=@(x,y) pi/x/2^y;
f22=@(x,y) x-y:y:x+y;

f1=@(x) exp(1i*pi*(0:Nr-1)*sin(x));
f2=@(x) 1/sqrt(Nt)*exp(1i*pi*(0:Nt-1)*sin(x));
f3=@(x) [real(x) -imag(x);imag(x) real(x)];
% f4=@(x) (f3(kron(X.'*conj(f2(x(2)).'),f1(x(1)).'))*[x(3);x(4)]-lambda)/sigma*sqrt(2);
% fun1=@(x) sum(-log(0.5*erfc(-z.*f4(x)/sqrt(2))));

deep=0;
Deep=8;
fmin1=Inf;
Pn=sigma^2;
alpha_r=0.5;
alpha_i=0.5;
while(deep<Deep)
    w1=f22(aoat,f11(Nr,deep));
    w2=f22(aodt,f11(Nt,deep));
%     w1=aoat-pi/Nr:pi/Nr/5:aoat+pi/Nr;
%     w2=aodt-pi/Nr:pi/Nr/5:aodt+pi/Nr;
%     tic
    for i=1:length(w1)
        for j=1:length(w2)
            %         x1=zeros(2,1);
            %         x0=ones(2,1);
%             x1(3)=x1(3)*sigma*2;
        x1=[alpha_r;alpha_i];
        x0=ones(2,1);
        gemma=f3(kron(X.'*conj(f2(w2(j)).'),f1(w1(i)).'));
        while(norm(x1-x0)>1e-5)
            x0=x1;
            h2=x0;
%             tic
            ys=(gemma*h2-lambda)/sigma*sqrt(2);
%             toc
%             tic
            tem=2./erfcx(-z.*ys/sqrt(2));
            temp=tem.*z;
            g=sum(temp.*gemma./sqrt(pi*Pn)).';
            po=(-tem.^2/pi/Pn+temp./sqrt(pi).*(-2*ys/sqrt(2))/Pn);
            H=gemma.'*(po.*gemma);
            x1=x0-H\g;
%             toc
        end
        temp=sum(-log(0.5*erfc(-z.*ys/sqrt(2))));
        if(temp<fmin1)
            fmin1=temp;
            aoat=w1(i);
            aodt=w2(j);
            alpha_r=x1(1);
            alpha_i=x1(2);
        end        
        end
    end
    deep=deep+1;
end
aoa=aoat;
aod=aodt;
% es_vec=[aoa aod alpha_r alpha_i];
% W1=pi/Nr/5;
% W2=pi/Nt/5;
% options=optimset('Display','off','tolX',1e-18);
% [s fmin2]=fmincon(fun1,es_vec,[],[],[],[],[es_vec(1)-W1 es_vec(2)-W2 -inf -inf],[es_vec(1)+W1 es_vec(2)+W2 inf inf],[],options);

loglihood=fmin1;