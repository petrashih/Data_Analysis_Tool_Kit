#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:24:31 2019

import projectNonOrthonormalBasis as pNOB

pNOB.nonOrthonormalProjection(targetVector, V)

@author: petrashih
"""
import numpy as np

def nonOrthonormalProjection(_targetVector, _V):    
    # norm of {vi}, [norm1, norm2, norm3, ..., normN]
    _normV = np.linalg.norm(_V.T, axis=1)
    
    # normalized basis set, [u1, u2, ..., uN]
    _normalizedV = _V / _normV
    
    _component = np.dot(_targetVector, _normalizedV.T[0]) * (_normalizedV.T[0])
    
    _projection = np.zeros(len(_V.T))
    _projection[0] = np.dot(_targetVector, _normalizedV.T[0])
    
    _residueVector = _targetVector
    for i in range(1,len(_V.T)):
        _residueVector = _residueVector - _component
        _component = None
        _component = np.dot(_residueVector, _normalizedV.T[i]) * (_normalizedV.T[i])
        _projection[i] = np.dot(_residueVector, _normalizedV.T[i])
#    print(_normalizedV * _projection)
#    print(sum((_normalizedV * _projection).T))
    return _projection, _normalizedV

# =============================================================================
# #### used for test
# targetVector = np.array([-5, 1, 2])
# # non-orthnormal basis set, [v1, v2, v3, ..., vN]
# V = np.array([[ 3,  0, 0,  0],
#               [ 0,  2, 0, 1],
#               [ 0,  1, 1, 0.5]])
# print(nonOrthonormalProjection(targetVector, V))
#     
# ####
# =============================================================================
