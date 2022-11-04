# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:13:10 2022

@author: Tib
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

N = 15         # max cavity population
D = 2          #number of atomic states?
geff = 1
ep=0.05*geff*2
wa = 0 # cavity and atom frequency
wc = 0
kappa = 0.05*geff        # cavity dissipation rate
gamma = 0        # atom dissipation rate
gamma_d = 1*kappa
LAMBDA =0*kappa

omega = 0.01*geff
alpha= 0
zeta=0

zeta = 0.01*geff
alpha= -2j*omega/kappa
omega= 0



system = t.MultiLevel(N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega, zeta, alpha)
H = system.hamiltonian(100)
c_ops = system.collapse()

# wlist = np.linspace(-1*np.pi *system.geff + system.wc, 1*np.pi *system.geff + system.wc, 200)
# spec = spectrum(H, wlist, c_ops, system.adagori, system.aori)

# fig, ax = plt.subplots()
# ax.plot(wlist, spec)
# ax.set_yscale('log')
# ax.set_ylabel(r'$S_a(\omega)$')
# ax.set_xlabel(r'$(\omega-\omega_0)/g_{eff}$')
# plt.title(r'$S_a(\omega)$ vs detuning for parameters: N=' + str(N) + r', D=' + str(D) + r', $g_{eff}$=' + str(geff)\
#           + r', $\epsilon$=' + str(ep)+ r', $\omega_a$=' + str(wa) + r', $\omega_c$=' + str(wc) +',\n' r'$\kappa$=' + str(kappa)\
#           + r',  $ \gamma$=' + str(gamma)+ r', $\gamma_d$=' + str(round(gamma_d,2)) + r', $\Lambda$=' + str(LAMBDA) + r', $\Omega$=' + str(omega) + r', $\zeta$='+str(zeta) + r', $\alpha$='+str(alpha), fontsize='small')

g2list = system.g2listcalc(system.a)
fig,ax=plt.subplots()
ax.plot(system.wl_list,g2list)
ax.set_yscale('log')
ax.set_ylabel(r'$g^{(2)}(0)$')
ax.set_xlabel(r'$(\omega_L-\omega_0)/g_{eff}$')
plt.title(r'$g^{(2)}(0)$ vs detuning for parameters: N=' + str(N) + r', D=' + str(D) + r', $g_{eff}$=' + str(geff)\
          + r', $\epsilon$=' + str(ep)+ r', $\omega_a$=' + str(wa) + r', $\omega_c$=' + str(wc) +',\n' r'$\kappa$=' + str(kappa)\
          + r',  $ \gamma$=' + str(gamma)+ r', $\gamma_d$=' + str(round(gamma_d,2)) + r', $\Lambda$=' + str(LAMBDA) + r', $\Omega$=' + str(omega) + r', $\zeta$='+str(zeta) + r', $\alpha$='+str(alpha), fontsize='small')

# # system.ss_dm(driving=True)
# darkstates = system.darkstate_proportion(driving=True)

# fig2,ax2 = plt.subplots()
# ax2.plot(system.wl_list,darkstates)
# ax2.set_ylabel(r'$p_{dark}$')
# ax2.set_xlabel(r'$(\omega_L-\omega_0)/g_{eff}$')
# plt.title(r'corresponding dark state proportions...', fontsize='small')