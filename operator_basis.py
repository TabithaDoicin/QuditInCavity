# -*- coding: utf-8 -*-
"""
Created on Wed Mar 9 12:40:51 2022

@author: Tib
"""


#Vee operators and basis definitions#

from qutip import *
import numpy as np

def vector2(d):
    out = np.empty([d,d],dtype=object)
    for n in range(d):
        for m in range(d):
            out[n,m] = basis(d,n)*basis(d,m).dag()
    return out
    
    
