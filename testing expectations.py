# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 17:20:27 2022

@author: Tib
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

N = 10         # max cavity population
D = 2          #number of atomic states?
geff = 1
ep=0.05*geff*2
wa = 0 # cavity and atom frequency
wc = 0
kappa = 0.05*geff        # cavity dissipation rate
gamma = 0        # atom dissipation rate
gamma_d = 0.2*kappa
LAMBDA =0.02*kappa

omega1 = 0.02*geff #having displacement makes p -> p_\alpha but also changes all ladders, which is why we specify!!! undisplaced 'original operators'
zeta1 = 0
alpha1 = -2j*omega1/kappa

omega2 = 0
zeta2 = -2j*omega1/kappa*geff #ladder operators not changed, so automatically Tr(p_\alpha a adag)
alpha2 = 0

sys1 = t.MultiLevel(N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega1, zeta1, alpha1)
sys2 = t.MultiLevel(N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega2, zeta2, alpha2)
sys1.hamiltonian()
sys1.collapse()
L1 = liouvillian(sys1.H,sys1.c_ops)
sys2.hamiltonian()
sys2.collapse()
L2 = liouvillian(sys2.H,sys2.c_ops)

expect1 = expect(sys1.adagori*sys1.aori ,sys1.ss_dm())
print(expect1)
expect2 =  expect(sys2.adag*sys2.a,sys2.ss_dm())
print(expect2)