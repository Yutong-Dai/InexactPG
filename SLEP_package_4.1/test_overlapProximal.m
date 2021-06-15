p=20;
g=5;

v=randn(p,1);

lambda1=0.0;
lambda2=1.1;
maxIter=2000;
tol=1e-12;

G=[1:8, 6:13, 8:15, 10:17, 13:20]-1;
W=[ [1 8 1]', [9 16 1]', [17 24 1]', [25 32 1]', [33 40 1]'];
W(1:2,:)=W(1:2,:)-1;


Y=zeros(length(G),1);
tic;
[x,gap, infor]=overlapping(v,  p, g, lambda1, lambda2,...
    W, G, Y, maxIter, 2, tol);
toc;