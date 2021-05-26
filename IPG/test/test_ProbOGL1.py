import sys
sys.path.append("../")

import numpy as np
from src.Problem import ProbOGL1
from src.regularizer import OGL1
from src.utils import GenOverlapGroup


class F:
    def __init__(self, n, p):
        self.n = n
        self.p = p


p = 7
f = F(100, p)
generator = GenOverlapGroup(p, 3, 5)
starts, ends = generator.get_group()
r = OGL1(Lambda=0.1, dim=p, starts=starts, ends=ends)
prob = ProbOGL1(f, r)
xk = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
gradfxk = 0.1 * np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
alphak = 0.2
params = {'eta': 0.1, 'xi': 0.5, 'subprob_maxiter': 100}
x, lambda_full, flag, iters = prob.ipg(xk, gradfxk, alphak, params)
print(x)
print(iters)
print(flag)
