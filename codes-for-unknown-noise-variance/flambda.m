function lambda=flambda(At,Dt,Rt,It,Nr,Nt,st)
ABS=exp(1i*2*pi*(0:Nr-1)*sin(At)/2);
AMS=1/sqrt(Nt)*exp(1i*2*pi*(0:Nt-1)*sin(Dt)/2);
temp=kron(st.'*conj(AMS.'),ABS.');
gemma=[real(temp) -imag(temp);imag(temp) real(temp)];
h2=[Rt;It];
lambda=gemma*h2;
