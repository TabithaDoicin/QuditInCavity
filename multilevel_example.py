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

N = 4         # number of cavity fock states
D = 2          #number of atomic states?
geff = 1
ep=0.05*geff*2
wa = 0 # cavity and atom frequency
wc = 0
kappa = 0.05*geff        # cavity dissipation rate
gamma = 0        # atom dissipation rate
gamma_d = 0*kappa
LAMBDA =0*kappa

omega = 0.02*geff
alpha= 0
zeta=0

zeta = -2j*omega/kappa*geff
omega=0
alpha=0#+2j*omega/kappa


system = t.MultiLevel(N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega, zeta, alpha)
Hlist = system.hamiltonian(300)
c_ops = system.collapse()

#wlist = np.linspace(-1*np.pi *system.geff + system.wc, 1*np.pi *system.geff + system.wc, 600)
#spec = spectrum(H, wlist, c_ops, system.adag, system.a)

#fig, ax = plt.subplots()
#ax.plot(wlist, np.log10(spec))

g2list = system.g2listcalc()
fig,ax=plt.subplots()
ax.plot(system.wl_list,g2list)
ax.set_yscale('log')
ax.set_ylabel(r'$g^{(2)}(0)$')
ax.set_xlabel(r'$(\omega_L-\omega_0)/g_{eff}$')
plt.title(r'$g^{(2)}(0)$ vs detuning for parameters: N=' + str(N) + r', D=' + str(D) + r', $g_{eff}$=' + str(geff)\
          + r', $\epsilon$=' + str(ep)+ r', $\omega_a$=' + str(wa) + r', $\omega_c$=' + str(wc) +',\n' r'$\kappa$=' + str(kappa)\
          + r',  $ \gamma$=' + str(gamma)+ r', $\gamma_d$=' + str(round(gamma_d,2)) + r', $\Lambda$=' + str(LAMBDA) + r', $\Omega$=' + str(omega) + r', $\zeta$='+str(zeta), fontsize='small')

# system.ss_dm(driving=True)
# darkstates = system.darkstate_proportion(driving=True)

# fig2,ax2 = plt.subplots()
# ax2.plot(system.wl_list,darkstates)
# ax2.set_ylabel(r'$p_{dark}$')
# ax2.set_xlabel(r'$(\omega_L-\omega_0)/g_{eff}$')
# plt.title(r'corresponding dark state proportions...', fontsize='small')