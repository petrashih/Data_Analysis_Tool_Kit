#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:11:54 2019

test for least square

@author: petrashih
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
## data points
#x = np.array([0, 1, 2, 3])
#y = np.array([-1, 0.2, 0.9, 2.1])
#A = np.vstack([x, np.ones(len(x)), np.ones(len(x))]).T
#m, n, c = np.linalg.lstsq(A, y, rcond=None)[0]
#print(m, n, c)
#


y = [3,2,3,4,3,4,5,4,5,4,4,5,4,5,4,5,5,5,4,5,4,3,4]

x1 = [
     [4,2,3,4,5,4,5,6,7,4,8,9,8,8,6,6,5,5,5,5,5,5,5],
     [4,1,2,3,4,5,6,7,5,8,7,8,7,8,7,8,7,7,7,7,7,6,5],
     [4,1,2,5,6,7,8,9,7,8,7,8,7,7,7,7,7,7,6,6,4,4,4]
     ]

## calculate the cross-correlation
def cross_correlation(f, g):
    ## calculate
    _correlation = np.average((f - np.average(f)) * (g - np.average(g)))
    _norm = np.std(f) * np.std(g)
    _correlation /= _norm
        
    return _correlation

def deviation_square(_a1, _a2):
    _de = (np.array(_a1) - np.array(_a2))**2
    return sum(_de)
##########################################################
## Start analysis

x = np.array(x1).T
x = sm.add_constant(x)
#y1 = y-np.average(y)
results = sm.OLS(endog=y, exog=x).fit()
#results = sm.OLS(endog=y, exog=x).fit_regularized(method='elastic_net', alpha=1.0, L1_wt=0.0)



######################
## Output results
# figure
plt.plot(y, 'o', label='Original data')
K = results.params[0]+results.params[1]*np.array(x1[0])+results.params[2]*np.array(x1[1])+results.params[3]*np.array(x1[2])
plt.plot(K, 'x', label='Fitted data')
plt.legend()

# print values
print("cross correlation = {} %".format(cross_correlation(y, K)*100))

print("deviation square = {}".format(deviation_square(K, y)))