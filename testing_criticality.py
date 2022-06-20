# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 18:39:51 2022

@author: Tib
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import simulation as t

N = 4                 # number of cavity fock states
wc = wa = 0 # cavity and atom frequency
g  = 1    # coupling strength
kappa = 0.5*g            # cavity dissipation rate
gamma = 0.5*kappa        # atom dissipation rate
gamma_d = 0.5*kappa
LAMBDA = 0.5*kappa
acc = 100
tacc = 100
omegalist = np.linspace(0,1,acc) #critical point at 0.5?
simlist = np.empty([acc],dtype=object)
tlist = np.linspace(0,100,tacc)
alist = np.empty([acc],dtype=object)
rho0 = tensor(coherent(N,3),basis(2,1))
fig,ax=plt.subplots()
for i in range(acc):
    simlist[i] = t.JC(N,wc,wa,g,kappa,gamma,gamma_d,LAMBDA,omegalist[i])
    simlist[i].hamiltonian()
    simlist[i].collapse()
    bonk = mesolve(simlist[i].H, rho0, tlist, c_ops=simlist[i].c_ops, e_ops=simlist[i].a, args=None, options=None, progress_bar=None, _safe_mode=True).expect[0]
    ax.plot(tlist,bonk)





