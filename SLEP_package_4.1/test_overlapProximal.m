clear, clc;
% current_path=cd;
% cd([current_path '/SLEP/CFiles/overlapping']);
% mex overlapping.c;
% cd(current_path);

p=10; % number of samples
g=3; % number of groups

xk = (0:9)' .* 0.1;
gradfxk =  xk .* 0.1;
alphak = 0.2;
uk = xk - alphak * gradfxk;
lambda1=0;
lambda2=alphak;
maxIter=100;
tol=1e-10;
flag = 0;



G=[0:4, 4:8,8:9];
W=[ [1 5 sqrt(5)]', [6 10 sqrt(5)]', [11 12 sqrt(2)]'];
W(1:2,:)=W(1:2,:)-1;

Y=zeros(length(G),1);
% tic;
fprintf("\n");
[x,gap, infor]=overlapping(uk,  p, g, lambda1, lambda2,...
    W, G, Y, maxIter, flag, tol);
% toc;
display(x')
display(gap)
display(infor(4))
