#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:36:50 2023

@author: tibbles
"""

import numpy as np
from qutip import *
import scipy as sp
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

N = 30             # number of cavity fock states #needs to be really high to properly classify eigenenergies
D1 = 3          #number of atomic states
D2 = 6
geff = 1
ep1=0.25
ep2=0.25
wa = 1            # cavity and atom frequency
wc = 1

#looking at geff variation
geff_list_min = 0
geff_list_max = 5
geff_list_num = 200

geff_list = np.linspace(geff_list_min, geff_list_max, geff_list_num)

system1_list = np.empty([geff_list_num], dtype = object)
system1_energies_list = np.empty([geff_list_num], dtype = object)

system2_list = np.empty([geff_list_num], dtype = object)
system2_energies_list = np.empty([geff_list_num], dtype = object)

for k in range(geff_list_num):
    system1_list[k] = t.MultiLevel(N, D1, geff_list[k], ep1, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=False)
    system1_list[k].hamiltonian(suppress=True)
    system1_energies_list[k] = np.array(system1_list[k].H.eigenenergies())
    
    system2_list[k] = t.MultiLevel(N, D2, geff_list[k], ep2, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=False)
    system2_list[k].hamiltonian(suppress=True)
    system2_energies_list[k] = np.array(system2_list[k].H.eigenenergies()) #ordered in g_eff
    
energy_sys1_list = np.empty([len(system1_energies_list[0])],dtype=object) #ordered in energy level
energy_sys2_list = np.empty([len(system2_energies_list[0])],dtype=object) 

for n in range(len(energy_sys1_list)): #the length is the same as N*D because hamiltonian diagonalisation is the amount of energy levels
    energy_sys1_list[n] = [system1_energies_list[k][n] for k in range(len(geff_list))]
    

for n in range(len(energy_sys2_list)): #the length is the same as N*D because hamiltonian diagonalisation is the amount of energy levels
    energy_sys2_list[n] = [system2_energies_list[k][n] for k in range(len(geff_list))]
    
fig, ax = plt.subplots()
additionscaling = np.empty([len(geff_list)])
plt.ylim(-2, 5)

for k in range(len(geff_list)):
    additionscaling[k] = (geff_list[k])**2 
    
for n in range(len(energy_sys1_list)):#plotting
    sys1_line, = ax.plot(geff_list, energy_sys1_list[n]+1*additionscaling, color = 'red', label='MQRM_sys1', linestyle = '-')

for n in range(len(energy_sys2_list)):#plotting
    sys2_line, = ax.plot(geff_list, energy_sys2_list[n]+1*additionscaling, color = 'blue', label='MQRM_sys2', linestyle = '-')
    
ax.set_ylabel(r'$(E+g_{eff}^2)/\omega$')
ax.set_xlabel(r'$g_{eff}/\omega$')
plt.title(r'Comparison of Energy Spectrums')
ax.legend(handles=[sys1_line, sys2_line])



