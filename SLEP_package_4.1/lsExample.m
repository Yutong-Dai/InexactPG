addpath('libsvm/')
loss = 'ls';
datasetName = 'abalone_scale';
filetypeConst
fileType = fileTypeDict(datasetName);
[X, Y] = set_up_xy(datasetName, fileType, './', true);
p = size(X,2);
num_grp = min(30, ceil(p * 0.3));
grp_size = fix(fix(p / num_grp) * 2);
blocks = gengroup(p, num_grp, grp_size);
weights = zeros(length(blocks),1);
for i=1:length(blocks)
    weights(i) = sqrt(length(blocks{i}));
end
sigma0 = normest(X)^2/size(X,1);
stop_par.tol_int = 1e-6;
stop_par.max_iter_ext = 10000;
stop_par.max_iter_int = 10000;
beta0 = zeros(p,1);
lambda0 = [];
tau = 0.1;
smooth_par = 0;
[beta,lambda,n_iter] = glopridu_algorithm(X,Y,blocks,tau,weights,smooth_par,beta0,lambda0,sigma0,stop_par);