'''
File: Problem.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-22 01:42
Last Modified: 2021-06-11 15:42
--------------------------------------------
Description:
'''
import sys
import warnings
sys.path.append("../")
from abc import ABC, abstractclassmethod
import numpy as np
from numba import jit
import src.utils as utils
from numpy.linalg import pinv
import src.latentOG.printUtils as printUtils
from scipy.sparse import csc_matrix

class Problem(ABC):
    def __init__(self, f, r) -> None:
        super().__init__()
        self.f = f
        self.r = r

    @abstractclassmethod
    def funcf(self, x):
        """
        function evaluation
        """
        pass

    @abstractclassmethod
    def funcr(self, x):
        """
        function evaluation
        """
        pass

    @abstractclassmethod
    def gradf(self, x):
        """
        gradient evaluation
        """
        pass

    @abstractclassmethod
    def ipg(self, x, gradfx, alpha):
        """
        return an inexact proximal gradient point
        """
        pass


class ProbLatentOG(Problem):
    def __init__(self, f, r) -> None:
        super().__init__(f, r)
        self.K = self.r.K
        self.n, self.p = self.f.n, self.f.p
        self.starts = self.r.starts
        self.ends = self.r.ends
        self.Lambda_group = self.r.Lambda_group.reshape(-1, 1)
        self.full_group_index = np.arange(self.K)
        self.M = np.sqrt(np.sum(self.Lambda_group ** 2))

    def funcf(self, x):
        return self.f.evaluate_function_value(x)

    def funcr(self, x, method='exact'):
        """
            Do not call it !!!!!!!
        """
        if method == 'exact':
            return self.r.func_exact(x)
        if method == 'ub':
            return self.r.func_ub(x)
        if method == 'lb':
            return self.r.func_lb(x)

    def gradf(self, x):
        # least square and logit loss doesn't require this
        return self.f.gradient()

    def _projectedGD(self, xk, gradfxk, alphak, params, outter_iter, lambda_init=None, outID=None):
        """
            perform projected gradient descent to solve the proximal problem
            return the approximation to prox_{alphak,r}(xk-alphak*gradfxk)

            termination type:
                params['inexact_type'] 1: ck||s_k||^2
                params['inexact_type'] 2: gamma_2
                params['inexact_type'] 3: O(1/k^3)
        reference: https://sites.math.washington.edu/~burke/crs/408/notes/nlp/gpa.pdf
        """
        self.fevals, self.gevals = 0, 0
        uk = xk - alphak * gradfxk
        # ----------------------- pre-screening to select variables to be projetced ---------
        to_be_projected_groups = np.full(self.K, False)
        for i in range(self.K):
            start, end = self.starts[i], self.ends[i]
            to_be_projected_groups[i] = utils.l2_norm(uk[start:end]) > alphak * self.Lambda_group[i]
        weights_proj_grp = alphak * self.Lambda_group[to_be_projected_groups]
        index_proj_grp = self.full_group_index[to_be_projected_groups]
        starts, ends = self.starts[to_be_projected_groups], self.ends[to_be_projected_groups]
        self.dualProbDim = len(index_proj_grp)
        if params['subsolver_verbose']:
            printUtils.print_subsolver_header(self.dualProbDim, params['subsolver'],
                                              params['inexact_type'], outter_iter, outID)
        # initialize the dual variable
        if lambda_init is None:
            lambda_working = np.zeros((self.K, 1))[index_proj_grp]
        else:
            lambda_working = lambda_init[index_proj_grp]
        lambda_full = np.zeros((self.K, 1))

        iters = 0
        correction = 0
        GA = len(index_proj_grp)
        if GA == 0:
            # in this case, the projection is equivalent to identity operator
            # z = xk - alphak * gradfxk
            # therefore, the proximal operator is evaluated exactly, which is 0
            x = np.zeros_like(xk)
            flag = 'exactsol'
            gap = epsilon = 0.0
            return x, lambda_full, flag, iters, gap, epsilon
        # I = np.zeros((self.p, GA))
        # set up the indicator functio 1{j\in g}
        # for j in range(GA):
        #     I[starts[j]:ends[j], j] = 1
        data, indices, indptr = create_I(starts, ends, GA)
        I = csc_matrix((data, indices, indptr), shape=(self.p, GA))
        # I_sparse = csc_matrix((data, indices, indptr), shape=(self.p, GA))
        # print("True I - Sparse I:")
        # print(np.sum(I - I_sparse.toarray()))
        # ---------------------------- Projected GD ---------------------------------
        while True:
            iters += 1
            # ----------------------- perform update -------------------------
            # calculate gradient of the dual objective
            s_working = I @ lambda_working
            denominator = 1 / (1 + s_working)
            grad = weights_proj_grp**2 - I.T @ ((uk * denominator)**2)
            self.gevals += 1
            norm_grad = utils.l2_norm(grad)
            # search direction
            d = np.maximum(0, lambda_working - params['projectedGD']['stepsize'] * grad) - lambda_working
            #  ---------------------------- backtrack line-search -------------------------------
            bak = 0
            stepsize = 1.0
            lambda_trial = np.zeros((GA, 1))
            dirder = grad.T @ d
            while True:
                lambda_trial = lambda_working + stepsize * d
                s_trial = I @ lambda_trial
                fdiff = (uk**2).T @ (I @ ((lambda_trial - lambda_working)) / ((1 + s_trial) * (1 + s_working))) + np.sum(weights_proj_grp**2 * (lambda_working - lambda_trial))
                self.fevals += 2
                # dirder = grad.T @ d
                rhs = params['eta'] * dirder * stepsize
                if np.abs(fdiff) <= 1e-18 and np.abs(rhs) <= 1e-18:
                    isContinue = False
                else:
                    isContinue = fdiff <= rhs
                if bak > 100:
                    # line search failed
                    flag = 'lnscfail'
                    gap = epsilon = 0.0
                    warnings.warn("Linesearch failed in projected GD")
                    return None, None, flag, iters, None, None
                if isContinue:
                    stepsize *= params['xi']
                    bak += 1
                else:
                    break
            norm_d = utils.l2_norm(d) * stepsize
            # ----------------------- check termination -------------------------------------
            if bak == 0:
                params['projectedGD']['stepsize'] *= 1.5
            else:
                params['projectedGD']['stepsize'] *= 2 / 3
            lambda_working = lambda_trial
            # check termination
            # case I: max number of iterations reached
            if iters > params['projectedGD']['maxiter']:
                flag = 'maxiters'
                warnings.warn("inexactness conditon is definitely not satisfied due to the maxiter limit.")
                break
            # case II: desidered termination conditon
            # feas_cond = np.all(grad[lambda_working == 0] >= 0)
            s_working = I @ lambda_working
            z = uk / (1 + s_working)  # hat_z_{k+1}
            z, violation = self._check_feasibility(z, alphak * self.Lambda_group)
            if violation:
                correction += 1
            x = uk - z  # hat_x_{k+1}
            if params['inexact_type'] == 1:
                sk = x - xk
                self.ck = (np.sqrt(6 / ((1 + params['gamma1']) * alphak)) - np.sqrt(2 / alphak)) ** 2 / 4
                epsilon = self.ck * (np.dot(sk.T, sk)[0][0])
                bb = utils.l2_norm(x - uk) / alphak + self.M
                aa = 0.5 / alphak
            elif params['inexact_type'] == 2:
                sk = x - xk
                epsilon = (1 - params['nu']) * (1 - params['gamma2']) * (np.dot(sk.T, sk)[0][0]) / (2 * alphak)
                bb = utils.l2_norm(x - uk) / alphak + self.M
                aa = (1 - (1 - 1 / params['nu']) * (1 - params['gamma2'])) / (2 * alphak)
            elif params['inexact_type'] == 3:
                epsilon = params['schimdt_const'] / (outter_iter ** params['delta'])
                bb = utils.l2_norm(x - uk) / alphak + self.M
                aa = 0.5 / alphak
            elif params['inexact_type'] == 4:
                sk = x - xk
                epsilon = (1 - params['nu']) * params['gamma4'] * (np.dot(sk.T, sk)[0][0]) / (2 * alphak)
                bb = utils.l2_norm(x - uk) / alphak + self.M
                aa = (1 - (1 - 1 / params['nu']) * params['gamma4']) / (2 * alphak)
            else:
                raise ValueError("not implemeted!")
            theta = (-bb + np.sqrt(bb**2 + 4 * aa * epsilon)) / (2 * aa)
            # duality gap
            primal, dual, gap = self._duality_gap(z, lambda_working, uk, s_working, weights_proj_grp)
            inexact_cond = (gap <= theta)
            if params['subsolver_verbose']:
                printUtils.print_subsolver_iterates(iters, norm_grad, params['projectedGD']['stepsize'], primal, dual, gap, theta,
                                                    bak, stepsize, norm_d, outID)
            # if feas_cond and inexact_cond:
            if inexact_cond:
                flag = 'desired '
                # print("======")
                break
        lambda_full[to_be_projected_groups] = lambda_working
        return x, lambda_full, flag, iters, gap, epsilon, theta, correction

    def _projectedNewton(self, xk, gradfxk, alphak, params, outter_iter,
                         lambda_init=None, outID=None):
        """
            perform projected newton to solve the proximal problem
            return the approximation to prox_{alphak,r}(xk-alphak*gradfxk)

            termination type:
                params['inexact_type'] 1: ck||s_k||^2
                params['inexact_type'] 2: gamma_2
                params['inexact_type'] 3: O(1/k^3)
        """
        uk = xk - alphak * gradfxk
        # ----------------------- pre-screening to select variables to be projetced ---------
        to_be_projected_groups = np.full(self.K, False)
        for i in range(self.K):
            start, end = self.starts[i], self.ends[i]
            to_be_projected_groups[i] = utils.l2_norm(uk[start:end]) > alphak * self.Lambda_group[i]
        weights_proj_grp = alphak * self.Lambda_group[to_be_projected_groups]
        index_proj_grp = self.full_group_index[to_be_projected_groups]
        starts, ends = self.starts[to_be_projected_groups], self.ends[to_be_projected_groups]
        self.dualProbDim = len(index_proj_grp)
        if params['subsolver_verbose']:
            printUtils.print_subsolver_header(self.dualProbDim, params['subsolver'],
                                              params['inexact_type'], outter_iter, outID)
        # initialize the dual variable
        if lambda_init is None:
            lambda_working = np.zeros((self.K, 1))[index_proj_grp]
        else:
            lambda_working = lambda_init[index_proj_grp]
        lambda_full = np.zeros((self.K, 1))

        iters = 0
        correction = 0
        GA = len(index_proj_grp)
        if GA == 0:
            # in this case, the projection is equivalent to identity operator
            # z = xk - alphak * gradfxk
            # therefore, the proximal operator is evaluated exactly, which is 0
            x = np.zeros_like(xk)
            flag = 'exactsol'
            gap = epsilon = 0.0
            return x, lambda_full, flag, iters, gap, epsilon
        I = np.zeros((self.p, GA))
        # set up the indicator functio 1{j\in g}
        for j in range(GA):
            I[starts[j]:ends[j], j] = 1
        # ---------------------------- Projected Newton ---------------------------------
        i_null = 0  # counter for early termination
        while True:
            iters += 1
            # ----------------------- perform update -------------------------
            # calculate gradient of the dual objective
            s_working = I @ lambda_working
            denominator = 1 / (1 + s_working)
            grad = weights_proj_grp**2 - I.T @ ((uk * denominator)**2)
            norm_grad = utils.l2_norm(grad)
            # --------------------- hand set eps=0.001 modify later ---------
            epsilon_enlargment = min(0.001, utils.l2_norm(lambda_working - np.maximum(0, lambda_working - grad)))
            tmp = 2 * (uk**2) * (denominator**3)

            # inactive: not in the I^+ set
            I_inactive = np.union1d(np.where(grad.reshape(-1) <= 0.0), np.where(lambda_working.reshape(-1) > epsilon_enlargment))
            n_inactive = len(I_inactive)
            # for early termination purpose
            if n_inactive != 0:
                i_null = 0
                G_inactive = np.zeros((n_inactive, n_inactive))
                for j in range(n_inactive):
                    idx_j = self.r.groups[index_proj_grp[I_inactive[j]]]
                    G_inactive[j, j] = np.sum(tmp[idx_j])
                    for s in range(j + 1, n_inactive):
                        idx_s = self.r.groups[index_proj_grp[I_inactive[s]]]
                        idx = np.intersect1d(idx_j, idx_s)
                        G_inactive[j, s] = np.sum(tmp[idx])
                        G_inactive[s, j] = G_inactive[j, s]
                p_inactive = pinv(G_inactive) @ grad[I_inactive]
            else:
                i_null += 1
                p_inactive = 0.0

            # active: in the I^+ set
            I_active = np.intersect1d(np.where(grad.reshape(-1) >= 0.0), np.where(lambda_working.reshape(-1) <= epsilon_enlargment))
            n_active = len(I_active)
            if n_active != 0:
                G_active = np.zeros((n_active, 1))
                for j in range(n_active):
                    idx_j = I_active[j]
                    G_active[j] = np.sum(tmp[idx_j])
                p_active = grad[I_active] / G_active
            else:
                p_active = 0.0

            #  ---------------------------- backtrack line-search -------------------------------
            dirder_inactive = grad[I_inactive].T @ p_inactive
            bak = 0
            stepsize = 1.0
            lambda_trial = np.zeros((GA, 1))
            while True:
                lambda_trial[I_active] = np.maximum(0, lambda_working[I_active] - stepsize * p_active)
                lambda_trial[I_inactive] = np.maximum(0, lambda_working[I_inactive] - stepsize * p_inactive)
                s_trial = I @ lambda_trial
                fdiff = (uk**2).T @ (I @ ((lambda_trial - lambda_working)) / ((1 + s_trial) * (1 + s_working))) + np.sum(weights_proj_grp**2 * (lambda_working - lambda_trial))
                dirder_active = grad[I_active].T @ (lambda_working[I_active] - lambda_trial[I_active])
                rhs = params['eta'] * (stepsize * dirder_inactive + dirder_active)
                if np.abs(fdiff) <= 1e-18 and np.abs(rhs) <= 1e-18:
                    isContinue = False
                else:
                    isContinue = fdiff < rhs
                if bak > 100:
                    # line search failed
                    flag = 'lnscfail'
                    gap = epsilon = 0.0
                    warnings.warn("Linesearch failed in projected Newton")
                    return None, None, flag, iters, None, None
                if isContinue:
                    stepsize *= params['xi']
                    bak += 1
                else:
                    break
            norm_d = utils.l2_norm(lambda_trial - lambda_working)
            # ----------------------- check termination -------------------------------------
            lambda_working = lambda_trial
            # check termination
            # case I: max number of iterations reached
            if iters > params['projectedNewton']['maxiter']:
                flag = 'maxiters'
                warnings.warn("inexactness conditon is definitely not satisfied due to the maxiter limit.")
                break
            # case II: solve full space projection for 5 consequitive iterations
            #          as suggested by authors, early termination
            if i_null == 5:
                warnings.warn("inexactness conditon may not satisfied due to the i_null=5.")
                flag = 'erlystop'
                break
            # case III: desidered termination conditon
            # feas_cond = np.all(grad[lambda_working == 0] >= 0)
            s_working = I @ lambda_working
            z = uk / (1 + s_working)  # hat_z_{k+1}
            z, violation = self._check_feasibility(z, alphak * self.Lambda_group)
            if violation:
                correction += 1
            x = uk - z  # hat_x_{k+1}
            if params['inexact_type'] == 1:
                sk = x - xk
                self.ck = (np.sqrt(6 / ((1 + params['gamma1']) * alphak)) - np.sqrt(2 / alphak)) ** 2 / 4
                epsilon = self.ck * (np.dot(sk.T, sk)[0][0])
                bb = utils.l2_norm(x - uk) / alphak + self.M
                aa = 0.5 / alphak
            elif params['inexact_type'] == 2:
                sk = x - xk
                epsilon = (1 - params['nu']) * (1 - params['gamma2']) * (np.dot(sk.T, sk)[0][0]) / (2 * alphak)
                bb = utils.l2_norm(x - uk) / alphak + self.M
                aa = (1 - (1 - 1 / params['nu']) * (1 - params['gamma2'])) / (2 * alphak)
            elif params['inexact_type'] == 3:
                epsilon = params['schimdt_const'] / (outter_iter ** params['delta'])
                bb = utils.l2_norm(x - uk) / alphak + self.M
                aa = 0.5 / alphak
            else:
                raise ValueError("not implemeted!")
            theta = (-bb + np.sqrt(bb**2 + 4 * aa * epsilon)) / (2 * aa)
            # duality gap
            primal, dual, gap = self._duality_gap(z, lambda_working, uk, s_working, weights_proj_grp)
            inexact_cond = (gap <= theta)
            if params['subsolver_verbose']:
                printUtils.print_subsolver_iterates(iters, norm_grad, '-', primal, dual, gap, theta, bak, stepsize, norm_d, outID)
            # if feas_cond and inexact_cond:
            if inexact_cond:
                flag = 'desired '
                # print("======")
                break
        lambda_full[to_be_projected_groups] = lambda_working
        return x, lambda_full, flag, iters, gap, epsilon, theta, correction

    def ipg(self, xk, gradfxk, alphak, params, outter_iter, lambda_init=None, outID=None):
        """
            inexactly solve the proximal graident problem
        """
        if params['subsolver'] == 'projectedNewton':
            return self._projectedNewton(xk, gradfxk, alphak, params, outter_iter, lambda_init, outID)
        elif params['subsolver'] == 'projectedGD':
            return self._projectedGD(xk, gradfxk, alphak, params, outter_iter, lambda_init, outID)
        else:
            raise ValueError(f"Unknown subsolver:{params['subsolver']}.")

    def _check_feasibility(self, z, bounds):
        """
            if z is in the set, return z itself
            otherwise return scaled version
        """
        violation = False
        for i in range(len(self.starts)):
            start, end = self.starts[i], self.ends[i]
            zg = z[start:end]
            zg_norm = np.sqrt(np.sum(zg * zg))
            if zg_norm > bounds[i]:
                violation = True
                z[start:end] = (bounds[i] / zg_norm) * zg
        return z, violation

    def _duality_gap(self, z, lambda_working, uk, s_working, weights_proj_grp):
        primal = utils.l2_norm(z - uk) ** 2
        dual = utils.l2_norm(uk)**2 - np.sum(weights_proj_grp**2 * lambda_working) - np.sum((uk**2 / (1 + s_working)))
        gap = primal - dual
        return primal, dual, gap

# @jit(nopython=True,cache=True)
def create_I(starts, ends, GA):
    indices = []
    indptr = [0]
    num_elements = 0
    for j in range(GA):
        num_elements +=  ends[j] - starts[j]
        indices += list(range(starts[j],ends[j]))
        indptr.append(num_elements)
    indices = np.array(indices)
    indptr = np.array(indptr)
    data = np.ones_like(indices)
    return data, indices, indptr