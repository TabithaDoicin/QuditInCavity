#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 00:44:22 2023

@author: Tabitha
"""

import numpy as np
from qutip import *
import scipy as sp
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

N = 10             # number of cavity fock states #needs to be really high to properly classify eigenenergies
D = 3             #number of atomic states
geff_forops = 1
ep=0.2*geff_forops
wa = 1            # cavity and atom frequency
wc = 1

#geff variation
geff_list_min = 0
geff_list_max = 1
geff_list_num = 100
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
    systems_gndstate_MBS_list[k] = systems_MBS_list[k].Udag * systems_MBS_list[k].H.groundstate()[1] #basis changed even needed for entropy?

entropy_cav_rwa = np.empty([geff_list_num], dtype= np.float128)
entropy_ato_rwa = np.empty([geff_list_num], dtype= np.float128)

entropy_cav_norwa = np.empty([geff_list_num], dtype= np.float128)
entropy_ato_norwa = np.empty([geff_list_num], dtype= np.float128)

entropy_cav_MBS = np.empty([geff_list_num], dtype= np.float128)
entropy_ato_MBS = np.empty([geff_list_num], dtype= np.float128)

for k in range(geff_list_num):
    rho_cav_rwa = ptrace(systems_gndstate_rwa_list[k], 0)
    rho_ato_rwa = ptrace(systems_gndstate_rwa_list[k], 1)
    entropy_cav_rwa[k] = entropy_vn(rho_cav_rwa)
    entropy_ato_rwa[k] = entropy_vn(rho_ato_rwa)
    
    rho_cav_norwa = ptrace(systems_gndstate_no_rwa_list[k], 0)
    rho_ato_norwa = ptrace(systems_gndstate_no_rwa_list[k], 1)
    entropy_cav_norwa[k] = entropy_vn(rho_cav_norwa)
    entropy_ato_norwa[k] = entropy_vn(rho_ato_norwa)
    
    rho_cav_MBS = ptrace(systems_gndstate_MBS_list[k], 0)
    rho_ato_MBS = ptrace(systems_gndstate_MBS_list[k], 1)
    entropy_cav_MBS[k] = entropy_vn(rho_cav_MBS)
    entropy_ato_MBS[k] = entropy_vn(rho_ato_MBS)

fig, ax = plt.subplots()
ax.set_ylabel(r'$S$')
ax.set_xlabel(r'$g_{eff}$')
MJCcav, = ax.plot(geff_list, entropy_cav_rwa, label='MJC', color='black')
MJCato, = ax.plot(geff_list, entropy_ato_rwa, label='MJC', color='black', linestyle = 'dotted')
MQRMcav, = ax.plot(geff_list, entropy_cav_norwa, color='red', label='MQRM')
MQRMato, = ax.plot(geff_list, entropy_ato_norwa, color='red', label='MQRM', linestyle = 'dotted')
MBScav, = ax.plot(geff_list, entropy_cav_MBS, color='blue', label='MBS_full')
MBSato, = ax.plot(geff_list, entropy_ato_MBS, color='blue', label='MBS_full', linestyle = 'dotted')

plt.xlim(geff_list_min,geff_list_max)
ax.legend(handles=[MJCcav, MJCato, MQRMcav, MQRMato, MBScav, MBSato])
    