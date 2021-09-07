% clear, clc;
% current_path=cd;
% cd([current_path '/SLEP/CFiles/overlapping']);
% mex overlapping.c;
% cd(current_path);
addpath(genpath('./SLEP'))
p=10; % number of samples
g=3; % number of groups

% xk = (0:9)' .* 0.1;
% gradfxk =  xk .* 0.1;
xk = [-1.1, -2.2, -3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11.4]';
gradfxk = 0.01.*[11.1, 2.2, 33.3, -44.4, -5.5, 36.6, 77.7, 8.8, 9.9, 11.4]';
alphak = 0.2;
uk = xk - alphak * gradfxk;
lambda1=0;
lambda2=alphak;
maxIter=100;
tol=1e-10;
flag = 2;



G=[0:4, 3:8,6:9];
W=[ [1 5 sqrt(1000)]', [6 11 sqrt(2)]', [11 15 sqrt(10)]'];
W(1:2,:)=W(1:2,:)-1;
W(3,:)=W(3,:).*2.0;

Y=zeros(length(G),1);
% tic;
fprintf("\n");
[x,gap, infor]=overlapping(uk,  p, g, lambda1, lambda2,...
    W, G, Y, maxIter, flag, tol);
% toc;
display(x')
display(gap)
display(infor(4))
