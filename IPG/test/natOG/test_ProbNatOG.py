import sys
import numpy as np
sys.path.append("../../")
from src.utils import GenOverlapGroup
from src.regularizer import natOG
from src.natOG.Problem import ProbNatOG

class F:
    def __init__(self, n, p):
        self.n = n
        self.p = p

p = 10
grp_size = 5
overlap_ratio = 0.2
generator = GenOverlapGroup(p, grp_size=grp_size,overlap_ratio=overlap_ratio)
starts, ends = generator.get_group()
Lambda = 1
r = natOG(Lambda, p, starts, ends)
r.createYStartsEnds()
# for i in range(r.K):
#     print(f"==group {i}==")
#     print(f' xstart:{r.starts[i]} | xend:{r.ends[i]} | Ystart:{r.Ystarts[i]} | Yend:{r.Yends[i]}')
    
f = F(100, p)
prob = ProbNatOG(f, r)
xk = np.arange(p).reshape(-1, 1) * 0.1
gradfxk = 0.1 * xk
# xk = np.array([-1.1, -2.2, -3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11.4]).reshape(-1, 1)
# gradfxk = np.array([11.1, 2.2, 33.3, -44.4, -5.5, 36.6, 77.7, 8.8, 9.9, 11.4]).reshape(-1, 1)
alphak = 0.2
params = {'eta': 0.1, 'xi': 0.5, 'subsolver': 'projectedGD', 'inexact_type': 1, 'gamma1': 1e-12}
params['projectedGD'] = {'maxiter':100, 'stepsize':1.0}
x, Y, flag, iters, gap = prob.ipg(xk, gradfxk, alphak, params, 1)
print(flag, iters, gap)
print(x.T)
