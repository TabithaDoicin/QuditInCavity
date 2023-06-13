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
import math

N = 600             # number of cavity fock states #needs to be really high to properly classify eigenenergies
M = 5             # number of atom states
geff = 0.2
wa = 1            # cavity and atom frequency
wc = 1

ep = 0.5

sys = t.MultiLevel(N, M, geff, ep, wc, wa, rwa=True)
H = sys.hamiltonian_nodriving()
eigspace = H.eigenstates()
sys_energies = eigspace[0]
sys_eigvecs = eigspace[1]
normalised_sys_eng_diff = t.elevelspacings(sys_energies,sys_eigvecs,sys.P,1,0.5)

fig, ax = plt.subplots()
####Plots of wigner-dyson and poisson distributions
spacings = np.linspace(0,8,200)
poissonian = np.exp(-spacings)
wignerdysonian = [np.pi * x/2 * math.e**(-(np.pi/4)*x**2) for x in spacings]
####
#plt.hist(sys_rwa_eng_diff, density=True, bins=100)
#plt.hist(sys_no_rwa_eng_diff, density=True, bins=100)
ax.plot(spacings,poissonian)
ax.plot(spacings,wignerdysonian)
plt.ylim([0,10])
plt.xlim([0,10])
plt.hist(normalised_sys_eng_diff, density=True, bins=40, log=False)
