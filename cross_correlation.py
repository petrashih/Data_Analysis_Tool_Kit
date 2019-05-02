#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:58:28 2019

@author: petrashih
"""

import numpy as np
import copy
import matplotlib.pyplot as plt


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

def objective(_c_arr):
    return cross_correlation(x.dot(_c_arr), y)


def deviation_square(_a1, _a2):
    _de = (np.array(_a1) - np.array(_a2))**2
    return sum(_de)
##########################################################
## Start analysis
x = np.array(x1).T
c_arr = np.random.rand(len(x1))
print(objective(c_arr))

max_pair = [objective(c_arr), copy.deepcopy(c_arr)]
change = np.zeros(len(x1))
scale = np.ones(len(x1)) * 30
grid = 50
epsilon = 0.001

lock = True
while lock:
    lock = False
    for m in range(len(x1)):
#        m = dominant - m - 1
        test = np.linspace(-scale[m], scale[m], grid)
        
        ## maximum in this iteration
        local_max_pair = copy.deepcopy(max_pair)
        for i in test.flat:
            new_carr = copy.deepcopy(max_pair[1])
            new_carr[m] += i      
            new_obj = objective(new_carr)
            if new_obj > local_max_pair[0]:
                local_max_pair = [new_obj, new_carr]
        if local_max_pair[0] > max_pair[0]:
            change[m] = (local_max_pair[1] - max_pair[1])[m]
            scale[m] = 5 * change[m] / np.linalg.norm(local_max_pair[1])
#            print('    ·update: a_{} changed from {:.2f} to {:.2f}'.format(m, max_pair[1][m], local_max_pair[1][m]))
#            print('    max increase from {:.2f} to {:.2f}'.format(max_pair[0], local_max_pair[0]))
            max_pair = local_max_pair
            lock = True
    total_change = np.linalg.norm(change)
    scale *= np.linalg.norm(max_pair[1]) / np.linalg.norm(scale)
    if total_change < epsilon or not lock:
        break
######################
######################
## Output results
# figure
#YY = (y - np.average(y)) / np.std(y) * np.std(new_f) + np.average(new_f)
plt.plot(y, 'o', label='Original data')

new_f = max_pair[1].dot(np.array(x1))
K = (new_f - np.average(new_f)) / np.std(new_f) * np.std(y) +np.average(y)
plt.plot(K, 'x', label='Fitted (shifted, scaled)')
plt.legend()

# print values
print("cross correlation = {} %".format(cross_correlation(y, K)*100))

print("deviation square = {}".format(deviation_square(K, y)))