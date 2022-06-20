# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:13:10 2022

@author: Tib
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import simulation as t

N = 3                 # number of cavity fock states
wc = wa = 0 # cavity and atom frequency
g  = 1    # coupling strength
kappa = g            # cavity dissipation rate
gamma = 0           # atom dissipation rate
gamma_d = 0*kappa
LAMBDA = 0
omega=0.01*g
system = t.JC(N, g, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega)
Hlist = system.hamiltonian(100,0,1)
qutip.settings.has_mkl = False
#print(H)
c_ops = system.collapse()

#wlist = np.linspace(-np.pi *system.g + system.wc, np.pi *system.g + system.wc, 600)
#spec = 1*spectrum(H, wlist, c_ops, system.adag, system.a)

#fig, ax = plt.subplots()
#ax.plot(wlist, np.log10(spec))


g2list = system.g2listcalc()
fig,ax=plt.subplots()
ax.plot(system.wl_list,np.log10(g2list))