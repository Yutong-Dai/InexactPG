'''
# File: problem.py
# Project: ipg
# Created Date: 2021-08-23 11:31
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-08-29 4:31
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''


import sys
sys.path.append('../')
import utils
import warnings
from numba import jit
import numpy as np
from numpy.linalg import norm
np.seterr(over='raise')


class Problem:
    def __init__(self, f, r):
        self.f = f
        self.r = r

    def funcf(self, x):
        return self.f.func(x)

    def funcr(self, x):
        return self.r.func(x)

    def gradf(self, x):
        return self.f.grad(x)




class ProbGL1:
    def __init__(self, f, r):
        super().__init__(f, r)
        self.K = self.r.K
        self.n, self.p = self.f.n, self.f.p
        self.starts = self.r.starts
        self.ends = self.r.ends
        self.Lambda_group = self.r.Lambda_group
