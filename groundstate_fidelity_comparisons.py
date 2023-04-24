#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:18:18 2023

@author: tibbles
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{physics}')

N = 50             # number of cavity fock states
D = 5             #number of atomic states
geff_forops = 1
ep=0.5*geff_forops
wa = 1            # cavity and atom frequency
wc = 1

#system for extracting operators (they are the same for rwa and no rwa)
sys = t.MultiLevel(N, D, geff_forops, ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)
#looking at geff variation
geff_list_min = 0
geff_list_max = 3
geff_list_num = 300
geff_list = np.linspace(geff_list_min, geff_list_max, geff_list_num)

systems_rwa_list = np.empty([geff_list_num], dtype = object)
systems_gndstate_rwa_list = np.empty([geff_list_num], dtype = object)

systems_no_rwa_list = np.empty([geff_list_num], dtype = object)
systems_gndstate_no_rwa_list = np.empty([geff_list_num], dtype = object)

systems_MBS_list = np.empty([geff_list_num], dtype = object)
systems_gndstate_MBS_list = np.empty([geff_list_num], dtype = object)

for k in range(geff_list_num):
    systems_rwa_list[k] = t.MultiLevel(N, D, geff_list[k], ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)
    systems_rwa_list[k].hamiltonian(suppress=True)
    systems_gndstate_rwa_list[k] = systems_rwa_list[k].H.groundstate()[1]
    
    systems_no_rwa_list[k] = t.MultiLevel(N, D, geff_list[k], ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=False)
    systems_no_rwa_list[k].hamiltonian(suppress=True)
    systems_gndstate_no_rwa_list[k] = systems_no_rwa_list[k].H.groundstate()[1]
    
    systems_MBS_list[k] = t.GeneralBlochSiegert(N, D, geff_list[k], ep, wc, wa)
    systems_MBS_list[k].hamiltonian()
    systems_gndstate_MBS_list[k] = systems_MBS_list[k].H.groundstate()[1]

fidelity_rwa_list = np.empty([geff_list_num], dtype = object)
fidelity_MBS_list = np.empty([geff_list_num], dtype = object)
interfidelity_list = np.empty([geff_list_num], dtype = object)

for k in range(geff_list_num):
    fidelity_rwa_list[k] = systems_gndstate_rwa_list[k].dag() * systems_gndstate_no_rwa_list[k]
    fidelity_MBS_list[k] = systems_gndstate_MBS_list[k].dag() * systems_gndstate_no_rwa_list[k]
    interfidelity_list[k] = systems_gndstate_MBS_list[k].dag() * systems_gndstate_rwa_list[k]
    
    fidelity_rwa_list[k] = np.abs(fidelity_rwa_list[k][0][0][0])**2
    fidelity_MBS_list[k] = np.abs(fidelity_MBS_list[k][0][0][0])**2
    interfidelity_list[k] = np.abs(interfidelity_list[k][0][0][0])**2
fig, ax = plt.subplots()
ax.plot(geff_list, fidelity_rwa_list)
ax.plot(geff_list, fidelity_MBS_list)
ax.plot(geff_list, interfidelity_list)