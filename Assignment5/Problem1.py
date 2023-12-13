#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:19:28 2023

@author: amaterasu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import pandas as pd

S = np.matrix(np.array([[0, 1, 0, 0, 1, 1],[1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]))

D = np.matrix(np.diag(np.sum(np.array(S), axis = 0)))

#print(D)

#1.1 

Y = np.matrix(np.array([0, 0, 0, -1, -1, 1])).T

#print(Y, "\n")


#1.2

S1 = sqrtm(np.linalg.matrix_power(D, -1)) @ S @ sqrtm(np.linalg.matrix_power(D, -1))

#print(S1)


class LabelSpreader:
    def __init__(self, S, Y, decay, max_iter=30):
        self.S = S
        self.Y = Y
        self.decay = decay
        self.max_iter = max_iter
    
    
    def print_label(self, Ylabel):
        for i, label in enumerate(Ylabel):
            print(f"v{i+1}: {label}")
    

    def label_spread(self):
        m = 0
        for i in range(self.max_iter):
            aS = self.decay * self.S
            m += np.linalg.matrix_power(aS, i) @ self.Y
        aS = self.decay * self.S
        F = (1 - self.decay) * m + np.linalg.matrix_power(aS, self.max_iter) @ self.Y
        return F

    def get_labels(self, F):
        Ylabel = ['' for _ in range(F.shape[0])]
        for i in range(F.shape[0]):
            if F[i] > 0:
                Ylabel[i] = "label 1"
            elif F[i] < 0 :
                Ylabel[i] = "label 2"
            else:
                Ylabel[i] = "No label"
        return Ylabel

    
            
#1.2
'''
spreader = LabelSpreader(S1, Y, 0.8, 1)

F = spreader.label_spread()
Ylabel = spreader.get_labels(F)
spreader.print_label(Ylabel)
'''
#1.3
'''
spreader = LabelSpreader(S1, Y, 0.8, 2)

F = spreader.label_spread()
Ylabel = spreader.get_labels(F)
spreader.print_label(Ylabel)
'''
#1.4
'''
spreader = LabelSpreader(S1, Y, 0.8, 999999)

F = spreader.label_spread()
Ylabel = spreader.get_labels(F)
spreader.print_label(Ylabel)
'''

S1 = (1-0.8)*np.linalg.matrix_power((np.identity(S1.shape[0])-0.8*S1),-1) @ Y
#print(S1)

L = D - S
#print(L)

Y = np.matrix([[0,1],[0,1],[1,0]])
#print(Y)

Fu = -1 * np.linalg.inv(L[0:3,0:3]) @ L[3:6,0:3] @ Y

print(Fu)

Yu = np.argmax(Fu,axis=1)
print(Yu)
