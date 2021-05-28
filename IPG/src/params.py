'''
File: params.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2019-10-31 15:51
Last Modified: 2021-05-27 22:43
--------------------------------------------
Description:
'''
from numpy import inf
params = {}
# termination
params['tol'] = 1e-6
params['max_iter'] = 100000
params['subprob_maxiter'] = 100
params['max_time'] = 3600
params['max_back'] = 100
# print
params['printlevel'] = 2
params['printevery'] = 20
# algorithm parameters
# try this no lager [1e-10, 1e-3]
params['eta'] = 1e-3
params['xi'] = 0.5
params['zeta'] = 0.8
# increase the alpha by a factor of beta
# this trick is used in tfocs implementation
# to boost numerical performance
params['beta'] = 1  # 1 / 0.9
params['update_alpha_strategy'] = 'none'  # model / none
params['optimality_measure'] = 'aprox'  # iterates
params['inexact_type'] = 1  # 1:mine 2: Lee 3:schimdt
params['gamma_1'] = 1e-12
params['gamma_2'] = 0.5
# parameters for schimdt method
params['delta'] = 1  # 1e-3
params['schimdt_const'] = 1  # c/k^3


fileTypeDict = {}
fileTypeDict['a9a'] = 'txt'
fileTypeDict['australian'] = 'txt'
fileTypeDict['breast_cancer'] = 'txt'
fileTypeDict['cod_rna'] = 'txt'
fileTypeDict['colon_cancer'] = 'bz2'
fileTypeDict['covtype'] = 'txt'
fileTypeDict['diabetes'] = 'txt'
fileTypeDict['duke'] = 'bz2'
fileTypeDict['fourclass'] = 'txt'
fileTypeDict['german_numer'] = 'txt'
fileTypeDict['gisette'] = 'bz2'
fileTypeDict['heart'] = 'txt'
fileTypeDict['ijcnn1'] = 'txt'
fileTypeDict['ionosphere'] = 'txt'
fileTypeDict['leu'] = 'bz2'
fileTypeDict['liver_disorders'] = 'txt'
fileTypeDict['madelon'] = 'txt'
fileTypeDict['mushrooms'] = 'txt'
fileTypeDict['phishing'] = 'txt'
fileTypeDict['rcv1'] = 'txt'
fileTypeDict['skin_nonskin'] = 'txt'
fileTypeDict['sonar'] = 'txt'
fileTypeDict['splice'] = 'txt'
fileTypeDict['svmguide1'] = 'txt'
fileTypeDict['svmguide3'] = 'txt'
fileTypeDict['w8a'] = 'txt'
fileTypeDict['avazu_app_tr'] = 'txt'
fileTypeDict['HIGGS'] = 'txt'
fileTypeDict['news20'] = 'txt'
fileTypeDict['real-sim'] = 'txt'
fileTypeDict['SUSY'] = 'txt'
fileTypeDict['url_combined'] = 'txt'
fileTypeDict['bodyfat_scale'] = 'txt'
fileTypeDict['bodyfat_scale_expanded7'] = 'mat'
fileTypeDict['bodyfat_scale_expanded2'] = 'mat'
fileTypeDict['bodyfat_scale_expanded1'] = 'mat'
fileTypeDict['bodyfat_scale_expanded5'] = 'mat'
fileTypeDict['YearPredictionMSD.t_expanded1'] = 'mat'
fileTypeDict['space_ga_scale_expanded1'] = 'mat'
fileTypeDict['space_ga_scale_expanded5'] = 'mat'
fileTypeDict['YearPredictionMSD.t'] = 'bz2'

# regression
fileTypeDict['abalone_scale'] = 'txt'
fileTypeDict['bodyfat_scale'] = 'txt'
fileTypeDict['cadata'] = 'txt'
fileTypeDict['cpusmall_scale'] = 'txt'
fileTypeDict['eunite2001'] = 'txt'
fileTypeDict['housing_scale'] = 'txt'
fileTypeDict['mg_scale'] = 'txt'
fileTypeDict['mpg_scale'] = 'txt'
fileTypeDict['pyrim_scale'] = 'txt'
fileTypeDict['space_ga_scale'] = 'txt'
fileTypeDict['E2006.train'] = 'txt'
fileTypeDict['log1p.E2006.train'] = 'txt'
fileTypeDict['YearPredictionMSD'] = 'txt'
fileTypeDict['blogData_train'] = 'txt'
fileTypeDict['UJIIndoorLoc'] = 'txt'
fileTypeDict['driftData'] = 'txt'
fileTypeDict['virusShare'] = 'txt'
fileTypeDict['triazines_scale'] = 'txt'


fileTypeDict['bodyfat_scale_expanded7'] = 'mat'
fileTypeDict['pyrim_scale_expanded5'] = 'mat'
fileTypeDict['triazines_scale_expanded4'] = 'mat'
fileTypeDict['housing_scale_expanded7'] = 'mat'

gammas = {}
gammas['bodyfat_scale_expanded7'] = [1e-4, 1e-5, 1e-6]
gammas['pyrim_scale_expanded5'] = [1e-2, 1e-3, 1e-4]
gammas['triazines_scale_expanded4'] = [1e-2, 1e-3, 1e-4]
gammas['housing_scale_expanded7'] = [1e-2, 1e-3, 1e-4]
groups = {}
groups['bodyfat_scale_expanded7'] = 388
groups['pyrim_scale_expanded5'] = 671
groups['triazines_scale_expanded4'] = 2118
groups['housing_scale_expanded7'] = 258
