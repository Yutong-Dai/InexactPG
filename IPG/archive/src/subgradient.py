'''
File: subgradient.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-04-16 00:42
Last Modified: 2021-04-17 21:40
--------------------------------------------
Description:
'''

class SubgradientProxGroupL1:
    def __init__(self, xk, alphak, gradfk):
        self.gradient_step = xk - alphak * gradfk