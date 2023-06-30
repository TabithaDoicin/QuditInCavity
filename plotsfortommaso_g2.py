#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:28:18 2023

@author: tibbles
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

plt.close()

N = 3         # number of cavity fock states
D1 = 2          #number of atomic states
D2 = 3
D3 = 3
geff = 1
ep=0.05*geff
wa = 0 # cavity and atom frequency
wc = 0
kappa = 0.05*geff        # cavity dissipation rate
gamma = 0        # atom dissipation rate
gamma_d = 1*kappa
LAMBDA =0*kappa

omega=0.01*geff
zeta=0
alpha=0

granularity = 200

system1 = t.MultiLevel(N, D1, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega, zeta, alpha, rwa=True)
system2 = t.MultiLevel(N, D2, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega, zeta, alpha, rwa=True)
system3 = t.Dicke(N, D3, geff, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega, tc=True)
H1 = system1.hamiltonian(granularity)
H2 = system2.hamiltonian(granularity)

system3.hamiltonian()
H3 = system3.hamiltonian_withdriving(-1*np.pi, 1*np.pi, granularity)

c_ops1 = system1.collapse()
c_ops2 = system2.collapse()
c_ops3 = system3.collapse()

g2list1 = system1.g2listcalc(system1.a)
g2list2 = system2.g2listcalc(system2.a)
g2list3 = system3.g2listcalc(system3.a)

fig, ax = plt.subplots()
ax.plot(system3.wl_list, g2list3, linewidth = 1, ls = '--', color = 'green')
ax.plot(system2.wl_list, g2list2, linewidth = 1.1, ls = '-', color = 'magenta')
ax.plot(system1.wl_list, g2list1, linewidth = 1, ls = '--', color = 'black')


left, width = 0, 1
bottom, height = 0, 1
right = left + width
top = bottom + height
p = plt.Rectangle((left, bottom), width, height, fill=False)
p.set_transform(ax.transAxes)
p.set_clip_on(False)
ax.add_patch(p)

plt.rc('font', size=12)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

ax.set_yscale('log')
ax.set_ylabel(r'$g^{(2)}(0)$')
ax.set_xlabel(r'$(\omega_{L}-\omega_0)/g_{eff}$')

