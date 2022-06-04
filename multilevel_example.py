# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:13:10 2022

@author: Tib
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import tufarellisys as t

N = 5         # number of cavity fock states
D = 4          #number of atomic states?
geff = 1
ep=0.1*geff
glist = np.linspace(geff/np.sqrt(D-1),geff/np.sqrt(D-1),D-1)
delta = np.linspace(-ep,ep,D-1)#for D=2 picks -ep/2 avoid confusion!!!! (use ep=0 for small D=2 (jc))
wa = 0 # cavity and atom frequency
wc = 0
kappa = geff*0.05        # cavity dissipation rate
gamma = 0        # atom dissipation rate
gamma_d = 0.2 *kappa
LAMBDA =0.02*kappa
omega = 0

system = t.MultiLevel(N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega)
H = system.hamiltonian()
c_ops = system.collapse()

wlist = np.linspace(-1*np.pi *system.geff + system.wc, 1*np.pi *system.geff + system.wc, 600)
spec = 1*spectrum(H, wlist, c_ops, system.adag, system.a)

fig, ax = plt.subplots()
ax.plot(wlist, np.log10(spec))

#ax.plot(np.imag(w))