import sys
sys.path.append("../")

import numpy as np
from src.exact.Problem import ProbOGL1
from src.regularizer import OGL1
from src.utils import GenOverlapGroup


class F:
    def __init__(self, n, p):
        self.n = n
        self.p = p

# non-overlapping test
# p = 7
# f = F(100, p)
# generator = GenOverlapGroup(p, 3, 5)
# # starts, ends = generator.get_group()
# # r = OGL1(Lambda=5, dim=p, starts=starts, ends=ends)
# starts, ends = [0, 3, 5], [2, 4, 6]
# r = OGL1(Lambda=20, dim=p, starts=starts, ends=ends)
# prob = ProbOGL1(f, r)
# xk = 1.0 * np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
# gradfxk = 0.1 * np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
# alphak = 0.2
# params = {'eta': 0.1, 'xi': 0.5, 'subprob_maxiter': 8, 'inexact_type': 1, 'gamma1': 1e-12}
# x, lambda_full, flag, iters = prob.ipg(xk, gradfxk, alphak, params)
# print(x.T)
# print(iters)
# print(flag)

# p = 7
# f = F(100, p)
# generator = GenOverlapGroup(p, 3, 5)
# starts, ends = generator.get_group()
# for tau in [0.001, 0.01, 0.1, 1, 5, 7, 10]:
#     r = OGL1(Lambda=tau, dim=p, starts=starts, ends=ends)
#     prob = ProbOGL1(f, r)
#     xk = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
#     gradfxk = 0.1 * np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
#     alphak = 0.2
#     params = {'eta': 0.1, 'xi': 0.5, 'subprob_maxiter': 100}
#     x, lambda_full, flag, iters = prob.ipg(xk, gradfxk, alphak, params)
#     print(x.T)
#     print(iters)
#     print(flag)


p = 1000
grp_size = 10
num_grp = 300
f = F(100, p)
generator = GenOverlapGroup(p, num_grp, grp_size)
starts, ends = generator.get_group()
r = OGL1(Lambda=10, dim=p, starts=starts, ends=ends)
prob = ProbOGL1(f, r)
# np.random.seed(10)
# xk = np.random.randn(p, 1)
xk = np.arange(p).reshape(-1, 1) * 0.1
gradfxk = 0.1 * xk
alphak = 0.2
params = {'eta': 0.1, 'xi': 0.5, 'subprob_maxiter': 100, 'inexact_type': 1, 'gamma1': 1e-12}
x, lambda_full, flag, iters = prob.ipg(xk, gradfxk, alphak, params)
print(iters)
print(flag)
