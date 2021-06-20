addpath('libsvm/');
addpath('SLEP/functions/overlapping/');
addpath('SLEP/opts/');
addpath('SLEP/CFiles/overlapping/');
loss = 'logit';
datasetName = 'diabetes';
filetypeConst
fileType = fileTypeDict(datasetName);
[X, Y] = set_up_xy(datasetName, fileType, '../../GroupFaRSA/db', true);
p = size(X,2);
grpsize = min(20, floor(p/2));
overlap_ratio = 0.1;
[G,W] = gengroup(p, grpsize, overlap_ratio);
G = G + 1;
W(1,:) = W(1,:) + 1;
W(2,:) = W(2,:) + 1;
%----------------------- Set optional items -----------------------
opts=[];

% Starting point
opts.init=2;        % starting from a zero point

% Termination 
opts.tFlag=3;       % the relative change is less than opts.tol
opts.maxIter=5000;  % maximum number of iterations
opts.tol=1e-5;      % the tolerance parameter

% regularization
opts.rFlag=0;       % use ratio

% Normalization
opts.nFlag=0;       % without normalization
opts.G=G;
opts.ind=W;


opts.rStartNum=100;

%----------------------- Run the code  -----------------------

z=[0, 0.01];

opts.maxIter2=1000;
opts.tol2=1e-8;
opts.flag2=2;
tic;
[x, c, funVal, ValueL]= overlapping_LogisticR(X,Y, z, opts);
toc;