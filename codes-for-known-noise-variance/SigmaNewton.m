function es_vec=SigmaNewton(Nr,Nt,st,lambda,ym,Pn)
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
        x1=zeros(2,1);
        x0=ones(2,1);
        gemma=f3(kron(st.'*conj(f2(w2(j)).'),f1(w1(i)).'));
        while(norm(x1-x0)>1e-10)
            x0=x1;
            h2=x0;
            ys=(gemma*h2-lambda)/sqrt(Pn/2);
            tem=2./erfcx(-ym.*ys/sqrt(2));
            temp=tem.*ym;
            g=sum(temp.*gemma./sqrt(pi*Pn)).';
            po=(-tem.^2/pi/Pn+temp./sqrt(pi).*(-2*ys/sqrt(2))/Pn);
            H=gemma.'*(po.*gemma);
            x1=x0-H\g;
        end
        temp=sum(-log(0.5*erfc(-ym.*ys/sqrt(2))));
        if(temp<fmin1)
            fmin1=temp;
            At(1)=w1(i);
            Dt(1)=w2(j);
            Rt(1)=x1(1);
            It(1)=x1(2);
        end
    end
end

w1=At-pi/N1:pi/10/N1:At+pi/N1;
w2=Dt-pi/N2:pi/10/N2:Dt+pi/N2;
fmin1=Inf;
for i=1:length(w1)
    for j=1:length(w2)
        x1=zeros(2,1);
        x0=ones(2,1);
        gemma=f3(kron(st.'*conj(f2(w2(j)).'),f1(w1(i)).'));
        while(norm(x1-x0)>1e-10)
            x0=x1;
            h2=x0;
            ys=(gemma*h2-lambda)/sqrt(Pn/2);
            tem=2./erfcx(-ym.*ys/sqrt(2));
            temp=tem.*ym;
            g=sum(temp.*gemma./sqrt(pi*Pn)).';
            po=(-tem.^2/pi/Pn+temp./sqrt(pi).*(-2*ys/sqrt(2))/Pn);
            H=gemma.'*(po.*gemma);
            x1=x0-H\g;
        end
        temp=sum(-log(0.5*erfc(-ym.*ys/sqrt(2))));
        if(temp<fmin1)
            fmin1=temp;
            At(1)=w1(i);
            Dt(1)=w2(j);
            Rt(1)=x1(1);
            It(1)=x1(2);
        end
    end
end

es_vec=[At Dt Rt Dt];