function es_vec=NoSigmaNewton(Nr,Nt,st,lambda,ym)
f1=@(x) exp(1i*2*pi*(0:Nr-1)*sin(x)/2);
f2=@(x) 1/sqrt(Nt)*exp(1i*2*pi*(0:Nt-1)*sin(x)/2);
f3=@(x) [real(x) -imag(x);imag(x) real(x)];

N1=Nr*2;
N2=Nt*2;

w1=-pi/2+0.01*pi:pi/N1:pi/2-0.01*pi;
w2=-pi/2+0.01*pi:pi/N2:pi/2-0.01*pi;

fmin1=Inf;
for i=1:length(w1)
    for j=1:length(w2)
%         x1=zeros(2,1);
%         x0=ones(2,1);
        x1=0.5*ones(3,1);
        x0=ones(3,1);
      
        gemma=f3(kron(st.'*conj(f2(w2(j)).'),f1(w1(i)).'));
      
        while(norm(x1-x0)>1e-10)
            x0=x1;
            h_bar=x0;
            alpha_bar=h_bar(1:2);
            sigma=h_bar(3);
            vect=[gemma -lambda];
            ys=sqrt(2)*(gemma*alpha_bar-sigma*lambda);
            
            tem=2./erfcx(-ym.*ys/sqrt(2));
            
            temp=tem.*ym;
            g=sum(temp.*vect./sqrt(pi)).';
            po=(-tem.^2/pi+temp./sqrt(pi).*(-sqrt(2)*ys));
            H=vect.'*(po.*vect);
            x1=x0-H\g;
        end
        
        temp=sum(-log(0.5*erfc(-ym.*ys/sqrt(2))));
        
        if(temp<fmin1)
            fmin1=temp;
            At=w1(i);
            Dt=w2(j);
            sigma=1/x1(3);
            Rt=x1(1)*sigma;
            It=x1(2)*sigma;
            xt=x1;
        end
      
    end
end

w1=At-pi/N1:pi/10/N1:At+pi/N1;
w2=Dt-pi/N2:pi/10/N2:Dt+pi/N2;

fmin1=Inf;
for i=1:length(w1)
    for j=1:length(w2)
%         x1=zeros(2,1);
%         x0=ones(2,1);
        x1=0.5*ones(3,1);
        x0=ones(3,1);
      
        gemma=f3(kron(st.'*conj(f2(w2(j)).'),f1(w1(i)).'));
      
        while(norm(x1-x0)>1e-10)
            x0=x1;
            h_bar=x0;
            alpha_bar=h_bar(1:2);
            sigma=h_bar(3);
            vect=[gemma -lambda];
            ys=sqrt(2)*(gemma*alpha_bar-sigma*lambda);
            
            tem=2./erfcx(-ym.*ys/sqrt(2));
            
            temp=tem.*ym;
            g=sum(temp.*vect./sqrt(pi)).';
            po=(-tem.^2/pi+temp./sqrt(pi).*(-sqrt(2)*ys));
            H=vect.'*(po.*vect);
            x1=x0-H\g;
        end
        
        temp=sum(-log(0.5*erfc(-ym.*ys/sqrt(2))));
        
        if(temp<fmin1)
            fmin1=temp;
            At=w1(i);
            Dt=w2(j);
            sigma=1/x1(3);
            Rt=x1(1)*sigma;
            It=x1(2)*sigma;
            xt=x1;
        end
      
    end
end

es_vec=[At Dt xt.'];