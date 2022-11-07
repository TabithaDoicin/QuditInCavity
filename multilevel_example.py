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


def main():
  N = 3         # number of cavity fock states
  D = 4          #number of atomic states
  geff = 1
  ep=0.1*geff
  wa = 0 # cavity and atom frequency
  wc = 0
  kappa = 1*geff        # cavity dissipation rate
  gamma = 0        # atom dissipation rate
  gamma_d = 0*kappa
  LAMBDA =0.01*kappa

  omega=0
  zeta=0
  alpha=0

  system = t.MultiLevel(N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega, zeta, alpha)
  H = system.hamiltonian()
  c_ops = system.collapse()

  wlist = np.linspace(-1*np.pi *system.geff + system.wc, 1*np.pi *system.geff + system.wc, 10000)
  spec = spectrum(H, wlist, c_ops, system.adag, system.a)

  fig, ax = plt.subplots()
  ax.plot(wlist, np.log10(spec), linewidth = 0.9)

  # g2list = system.g2listcalc()
  # fig,ax=plt.subplots()
  # ax.plot(system.wl_list,g2list)
  # ax.set_yscale('log')
  # ax.set_ylabel(r'$g^{(2)}(0)$')
  # ax.set_xlabel(r'$(\omega_L-\omega_0)/g_{eff}$')
  # plt.title(r'$g^{(2)}(0)$ vs detuning for parameters: N=' + str(N) + r', D=' + str(D) + r', $g_{eff}$=' + str(geff)\
  #           + r', $\epsilon$=' + str(ep)+ r', $\omega_a$=' + str(wa) + r', $\omega_c$=' + str(wc) +',\n' r'$\kappa$=' + str(kappa)\
  #           + r',  $ \gamma$=' + str(gamma)+ r', $\gamma_d$=' + str(round(gamma_d,2)) + r', $\Lambda$=' + str(LAMBDA) + r', $\Omega$=' + str(omega) + r', $\zeta$='+str(zeta), fontsize='small')

  # system = t.MultiLevel(N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega, zeta, alpha)
  # H = system.hamiltonian(100)
  # c_ops = system.collapse()

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

if __name__ == "__main__":
  __spec__ = None
  main()
