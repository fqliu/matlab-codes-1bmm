function [aoa aod alpha_r alpha_i sigma_v loglihood]=f_nosigma_newton_refine(aoat,aodt,Nr,Nt,K,z,X,lambda)


f11=@(x,y) pi/x/2^y;
f22=@(x,y) x-y:y:x+y;

f1=@(x) exp(1i*2*pi*(0:Nr-1)*sin(x)/2);
f2=@(x) 1/sqrt(Nt)*exp(1i*2*pi*(0:Nt-1)*sin(x)/2);
f3=@(x) [real(x) -imag(x);imag(x) real(x)];
% f5=@(x) (f3(kron(X.'*conj(f2(x(2)).'),f1(x(1)).'))*[x(3);x(4)]-x(5)*lambda)*sqrt(2);
% fun2=@(x) sum(-log(0.5*erfc(-z.*f5(x)/sqrt(2))));

deep=0;
Deep=8;
fmin1=Inf;
alpha_r=0.5;
alpha_i=0.5;
sigma=1;
while(deep<Deep)
    w1=f22(aoat,f11(Nr,deep));
    w2=f22(aodt,f11(Nt,deep));
%     w1=aoat-pi/Nr:pi/Nr/5:aoat+pi/Nr;
%     w2=aodt-pi/Nr:pi/Nr/5:aodt+pi/Nr;
    for i=1:length(w1)
        for j=1:length(w2)
            %         x1=zeros(2,1);
            %         x0=ones(2,1);
%             x1(3)=x1(3)*sigma*2;
            x1=[alpha_r;alpha_i;sigma];
            x0=ones(3,1);
            gemma=f3(kron(X.'*conj(f2(w2(j)).'),f1(w1(i)).'));
            index=0;
            while(norm(x1-x0)>1e-5)
                index=index+1;
                x0=x1;
                h_bar=x0;
                alpha_bar=h_bar(1:2);
                sigmat=h_bar(3);
                vect=[gemma -lambda];
                ys=sqrt(2)*(gemma*alpha_bar-sigmat*lambda);
                
                tem=2./erfcx(-z.*ys/sqrt(2));
                
                temp=tem.*z;
                g=sum(temp.*vect./sqrt(pi)).';
                po=(-tem.^2/pi+temp./sqrt(pi).*(-sqrt(2)*ys));
                H=vect.'*(po.*vect);
                x1=x0-H\g;
            end
            
            temp=sum(-log(0.5*erfc(-z.*ys/sqrt(2))));
            
            if(temp<fmin1)
                fmin1=temp;
                aoat=w1(i);
                aodt=w2(j);
                sigma=1/x1(3);
                alpha_r=x1(1)*sigma;
                alpha_i=x1(2)*sigma;
                sigma_v=sigma;
            end
        end
    end
    deep=deep+1;
end
% sigma_v=sigma;
aoa=aoat;
aod=aodt;
% es_vec=[aoa aod alpha_r alpha_i 1/sigma_v];
% W1=pi/Nr/10;
% W2=pi/Nt/10;
% options=optimset('Display','off','tolX',1e-18);
% [s fmin1]=fmincon(fun2,es_vec,[],[],[],[],[es_vec(1)-W1 es_vec(2)-W2 -inf -inf -inf],[es_vec(1)+W1 es_vec(2)+W2 inf inf inf],[],options);
% aoa=s(1);
% aod=s(2);
% sigma_v=1/s(5);
% alpha_r=s(3)*sigma_v;
% alpha_i=s(4)*sigma_v;

loglihood=fmin1;

