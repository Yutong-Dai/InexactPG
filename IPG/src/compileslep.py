'''
# File: test_slep.py
# Project: test
# Created Date: 2021-09-06 3:21
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2021-09-06 5:44
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''


import cppimport
import numpy as np
import cppimport
slep = cppimport.imp("slep")

# number of samples
p = 10
# number of groups
g = 3

xk = np.array([-1.1, -2.2, -3.3, 4.4, 5.5, 6.6,
               7.7, 8.8, 9.9, 11.4]).reshape(-1, 1)
gradfxk = 0.01 * np.array([11.1, 2.2, 33.3, -44.4, -
                           5.5, 36.6, 77.7, 8.8, 9.9, 11.4]).reshape(-1, 1)
alphak = 0.2
uk = xk - alphak * gradfxk
lambda1 = 0.0
lambda2 = alphak
maxIter = 100
tol = 1e-10
flag = 2


G = np.concatenate((np.arange(0, 5), np.arange(3, 9), np.arange(6, 10))) * 1.0
W = np.array([[0, 5, 10], [4, 10, 14], [
             np.sqrt(1000), np.sqrt(2), np.sqrt(10)]])
W[2, :] = W[2, :] * 2.0
w = W.T.reshape(-1, 1)
Y = np.zeros((len(G), 1))
gap = 0.0
info = np.zeros((5, 1))
proxgrad = np.zeros_like(uk)

# print(uk)
# print(p, g, lambda1, lambda2)
# print(W)
# print(G)
# print(Y.shape)
# print(w)

slep.overlapping_py(uk, p, g, lambda1, lambda2,
                    w, G, Y, maxIter, flag, tol, proxgrad, gap, info)
print('sol', proxgrad.T)
print('iters:', info[3])
