# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 21:22:51 2022

@author: Tib
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t



N = 3         # number of cavity fock states
D1 = 2          #number of atomic states
D2 = 4
D3 = 7
geff = 1
ep=0.4*geff
wa = 0 # cavity and atom frequency
wc = 0
kappa = 1*geff        # cavity dissipation rate
gamma = 0        # atom dissipation rate
gamma_d = 0*kappa
LAMBDA =0.01*kappa

omega=0
zeta=0
alpha=0

granularity = 50000

system1 = t.MultiLevel(N, D1, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega, zeta, alpha)
system2 = t.MultiLevel(N, D2, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega, zeta, alpha)
system3 = t.MultiLevel(N, D3, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega, zeta, alpha)
H1 = system1.hamiltonian()
H2 = system2.hamiltonian()
H3 = system3.hamiltonian()
c_ops1 = system1.collapse()
c_ops2 = system2.collapse()
c_ops3 = system3.collapse()

wlist1 = np.linspace(-1*np.pi *system1.geff + system1.wc, 1*np.pi *system1.geff + system1.wc, granularity)
wlist2 = np.linspace(-1*np.pi *system2.geff + system2.wc, 1*np.pi *system2.geff + system2.wc, granularity)
wlist3 = np.linspace(-1*np.pi *system3.geff + system3.wc, 1*np.pi *system3.geff + system3.wc, granularity)

system1.ss_dm()
system2.ss_dm()
system3.ss_dm()
pdark1 = system1.darkstate_proportion()
pdark2 = system2.darkstate_proportion()
pdark3 = system3.darkstate_proportion()
print(pdark2,pdark3)
spec1 = spectrum(H1, wlist1, c_ops1, system1.adag, system1.a)
spec2 = 1/(1-pdark2) * spectrum(H2, wlist2, c_ops2, system2.adag, system2.a)
spec3 = 1/(1-pdark3) * spectrum(H3, wlist3, c_ops3, system3.adag, system3.a)


fig, ax = plt.subplots()
ax.plot(wlist3, spec3, linewidth = 1, color = 'magenta')
ax.plot(wlist2, spec2, linewidth = 0.9, ls = '--', color = 'darkblue')
ax.plot(wlist1, spec1, linewidth = 0.9, ls = '--', color = 'darkred')

plt.text(-0.62, 1.9e-4, r'$p_{dark} = $' + str(round(pdark3,3)), fontsize = 10, 
         bbox = dict(boxstyle='roundtooth', fc="w", ec="magenta"))
plt.text(-0.62, 0.8e-4, r'$p_{dark} = $' + str(round(pdark2,3)), fontsize = 10, 
         bbox = dict(boxstyle='roundtooth', fc="w", ec="darkblue"))

ax.set_yscale('log')
ax.set_ylabel(r'$S(\omega)$')
ax.set_xlabel(r'$(\omega-\omega_0)/g_{eff}$')

