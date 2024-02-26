#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:34:59 2024

@author: Tabitha
"""

import numpy as np
from qutip import *
import scipy as sp
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

N = 200             # number of cavity fock states #needs to be really high to properly classify eigenenergies
D = 10          #number of atomic states
geff = 1
ep=0*geff
wa = 1            # cavity and atom frequency
wc = 1
C = 2500
crit = 0.5

geff_list_min = 0.7*0.5
geff_list_max = 1.3*0.5
geff_list_num = 100

geff_list = np.linspace(geff_list_min, geff_list_max, geff_list_num)

systems_list = np.empty([geff_list_num], dtype = object)
systems_energies_list = np.empty([geff_list_num], dtype = object)

for k in range(geff_list_num):
    systems_list[k] = t.HighMultilevel(N, D, geff_list[k], ep, wc, wa, C, crit)
    systems_list[k].hamiltonian()
    systems_energies_list[k] = np.array(systems_list[k].H.eigenenergies()) 
    systems_energies_list[k] = systems_energies_list[k] - systems_energies_list[k].min()
    
energy_list = np.empty([len(systems_energies_list[0])],dtype=object) #energy levels specifically!!
for n in range(len(energy_list)): #the length is the same as N*D because hamiltonian diagonalisation is the amount of energy levels
    energy_list[n] = [systems_energies_list[k][n] for k in range(len(geff_list))]

fig, ax = plt.subplots()
plt.ylim(0, 0.5)
plt.xlim(geff_list_min,geff_list_max)
for n in range(50):#plotting
    ELevelLines, = ax.plot(geff_list,energy_list[n], color = 'black', linestyle = '-', label = 'energies') #no rescaling?