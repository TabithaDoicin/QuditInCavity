# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:13:10 2022

@author: Tib
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import simulation as t

N = 5         # number of cavity fock states
D = 4          #number of atomic states?
geff = 1
ep=0.05*geff*2
wa = 0 # cavity and atom frequency
wc = 0
kappa = 0.05*geff        # cavity dissipation rate
gamma = 0        # atom dissipation rate
gamma_d = 0.2*kappa
LAMBDA =0*kappa
omega = 0.1*geff
zeta = 0*geff
qutip.settings.has_mkl = False
system = t.MultiLevel(N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega, zeta)
Hlist = system.hamiltonian(200)
c_ops = system.collapse()

#wlist = np.linspace(-1*np.pi *system.geff + system.wc, 1*np.pi *system.geff + system.wc, 600)
#spec = spectrum(H, wlist, c_ops, system.adag, system.a)

#fig, ax = plt.subplots()
#ax.plot(wlist, np.log10(spec))

g2list = system.g2listcalc()
fig,ax=plt.subplots()
ax.plot(system.wl_list,np.log10(g2list))

system.ss_dm(driving=True)
darkstates = system.darkstate_proportion(driving=True)

fig2,ax2 = plt.subplots()
ax2.plot(system.wl_list,darkstates)