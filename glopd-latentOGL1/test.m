% d = 7; grpsize=5;
% blocks = {[1:5]',[2:6]',[3:7]'};
% xk = [1:d]';
% gradfxk = 0.1.* [1:d]';
% alphak = 0.2;
% uk = xk - alphak .* gradfxk;

% taus = [0.001, 0.01,0.1,1,5,7, 10];
% for i=1:length(taus)
%     tau = taus(i);
%     weights = [sqrt(grpsize),sqrt(grpsize),sqrt(grpsize)]';
%     lambda0 = zeros(3,1);
%     tol=1e-6;max_iter=100;
%     [w,q,lambda_tot] = glo_prox(uk,tau,blocks,weights,lambda0,tol,max_iter);
%     display(w')
%     display(q)
%     fprintf('=====\n')
% end

% tau = 5;
% weights = [sqrt(grpsize),sqrt(grpsize),sqrt(grpsize)]';
% lambda0 = zeros(3,1);
% tol=1e-12;max_iter=100;
% [w,q,lambda_tot] = glo_prox(uk,tau,blocks,weights,lambda0,tol,max_iter);
% display(w')
% display(q)

d=70;
blocks = {[1:30]',[14:50]',[15:70]'};
xk = 1.0 .* [1:d]';
gradfxk = 0.1.* [1:d]';
alphak = 0.2;
uk = xk - alphak .* gradfxk;

lambda = 0.2;
weights = lambda .* [length(blocks{1}),length(blocks{2}),length(blocks{3})]';
lambda0 = zeros(3,1);
tol=1e-3;max_iter=100;
[w,q,lambda_tot,z] = glo_prox(uk,alphak,blocks,weights,lambda0,tol,max_iter);
display(w')
display(q)
isfeasible(z, alphak,blocks, weights)