function h = func_unqt_ML(Y, X, Nr, Nt)

H=Y * X' / (X * X');

h = reshape(H, Nr*Nt, 1);
h = [real(h); imag(h)];