#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 02:50:30 2019

@author: petrashih
"""
import numpy as np

def gram_schmidt_columns(_X):
    _Q, _R = np.linalg.qr(_X)
    return _Q

# test
A = np.array([[1, 2, -1],[1, -1, 2],[-1, 1, 1],[1, -1, 2]])
print("A = {}".format(A))

print("Q = {}".format(gram_schmidt_columns(A)))