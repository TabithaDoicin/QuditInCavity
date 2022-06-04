# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:13:10 2022

@author: Tib
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import tufarellisys as t

N = 3                 # number of cavity fock states
wc = wa = 0 # cavity and atom frequency
g  = 1    # coupling strength
kappa = g*0.05            # cavity dissipation rate
gamma = 0           # atom dissipation rate
gamma_d = 0.01*kappa
LAMBDA = 0.01*kappa
omega=0

system = t.JC(N, g, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega)
H= system.hamiltonian()
print(H)
c_ops = system.collapse()

wlist = np.linspace(-np.pi *system.g + system.wc, np.pi *system.g + system.wc, 600)
spec = 1*spectrum(H, wlist, c_ops, system.adag, system.a)

fig, ax = plt.subplots()
ax.plot(wlist, np.log10(spec))