#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:05:33 2019

@author: petrashih
"""
import numpy as np
EigenVectors = [[ 1, 2, 3, 4, 5, 6],
               [ 7, 8, 9,10,11,12],
               [13,14,15,16,17,18],
               [19,20,21,22,23,24],
               [25,26,27,28,29,30],
               [31,32,33,34,35,36]]
EigenVectors = np.array(EigenVectors)

parameters = [1, 10, 100, 1000, 10000, 100000]
parameters = np.array(parameters)

Weighted_Vectors = EigenVectors[:,:6] * parameters

Collected_Mode = Weighted_Vectors.dot(np.ones(len(parameters)))

AtomVector = Collected_Mode.reshape(-1,3)