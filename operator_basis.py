# -*- coding: utf-8 -*-
"""
Created on Wed Mar 9 12:40:51 2022

@author: Tib
"""


#Vee operators and basis definitions#

from qutip import *
import numpy as np
import ggm

def vector(j,k,d): #0 is the ground state
    if j<k:
        return 0.5*(ggm.gellmann(k,j,d)+1j*ggm.gellmann(j,k,d))
    elif j>k:
        return 0.5*(ggm.gellmann(j,k,d)-1j*ggm.gellmann(k,j,d))
    elif j==k and j==1:
        return (1/d * np.identity(d) 
    + sum([1/np.sqrt(2*(j+n)*(j+n+1)) * ggm.gellmann(j+n,j+n,d) for n in range(d-j)]))
    elif j==k:
        return 1/d * np.identity(d) - np.sqrt((j-1)/2*j)*ggm.gellmann(j-1,j-1,d)
    + sum([1/np.sqrt(2*(j+n)*(j+n+1)) * ggm.gellmann(j+n,j+n,d) for n in range(d-j)])



