#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 03:56:25 2023

@author: tibbles
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t
import math

N =2500             # number of cavity fock states #needs to be really high to properly classify eigenenergies
M = 2             # number of atom states
geff = 2
wa = 1            # cavity and atom frequency
wc = 1

ep = 0.2

sys = t.MultiLevel(N, M, geff, ep, wc, wa, rwa=True)
H = sys.hamiltonian_nodriving()
eigspace = H.eigenstates()
sys_energies = eigspace[0]
sys_eigvecs = eigspace[1]
normalised_sys_eng_diff, espacings = t.elevelspacings(sys_energies, sys_eigvecs, sys.P, 1, 0.6)
n = np.linspace(0,len(espacings)-1,len(espacings))
fig, ax = plt.subplots()
ax.scatter(n,espacings)
plt.ylim([0,4])
plt.xlim([0,len(espacings)])