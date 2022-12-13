# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:44:37 2022

@author: Tib
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

N = 100             # number of cavity fock states #needs to be really high to properly classify eigenenergies
D = 5             #number of atomic states
geff = 0.2
ep=3*geff
wa = 1            # cavity and atom frequency
wc = 1

#system initialisations and hamiltonians and energy levels
sys_rwa = t.MultiLevel(N, D, geff, ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)
sys_rwa.hamiltonian()
sys_rwa_energies = np.array(sys_rwa.H.eigenenergies())
sys_rwa_eng_diff = [np.subtract(sys_rwa_energies[n+1], sys_rwa_energies[n]) for n in range(len(sys_rwa_energies)-1)]

sys_no_rwa = t.MultiLevel(N, D, geff, ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=False)
sys_no_rwa.hamiltonian()
sys_no_rwa_energies = np.array(sys_no_rwa.H.eigenenergies())
sys_no_rwa_eng_diff = [np.subtract(sys_no_rwa_energies[n+1], sys_no_rwa_energies[n]) for n in range(len(sys_no_rwa_energies)-1)]

sys_dicke = t.Dicke(N, D, geff, wc, wa) #dicke model
sys_dicke.hamiltonian()
sys_dicke_energies = np.array(sys_dicke.H.eigenenergies())
sys_dicke_eng_diff = [np.subtract(sys_dicke_energies[n+1], sys_dicke_energies[n]) for n in range(len(sys_dicke_energies)-1)]

plt.hist(sys_rwa_eng_diff, density=True, bins=100)
#plt.hist(sys_no_rwa_eng_diff, density=True, bins=100)
#plt.hist(sys_dicke_eng_diff, density=True, bins=100)