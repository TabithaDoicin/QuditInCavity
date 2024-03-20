#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:38:25 2024

@author: tibbles
"""

import numpy as np
from qutip import *
import scipy as sp
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import matplotlib
import simulation as t
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
N = 30             # number of cavity fock states #needs to be really high to properly classify eigenenergies
D = 4          #number of atomic states
geff = 1
ep=0.5*geff
wa = 1            # cavity and atom frequency
wc = 1

geff_list_min = 0
geff_list_max = 2.5
geff_list_num = 150

geff_list_iterator = np.linspace(geff_list_min,geff_list_max,geff_list_num)
deltalist = t.randomlydistribute(0, ep,D-1)
print(deltalist)
#gefflist_normalised = t.glist_generator(D-1,True)
position=2
dominance=0.95
gefflist_normalised = t.glist_gen_dominance(D-1, position, dominance)
print(gefflist_normalised)
geff_list = np.zeros([geff_list_num],dtype=object)
for k in range(len(geff_list)):
    geff_list[k] = geff_list_iterator[k]*gefflist_normalised
systems_list = np.empty([geff_list_num], dtype = object)
systems_energies_list = np.empty([geff_list_num], dtype = object)

epinput = ep #either ep or deltalist
for k in range(geff_list_num):
    systems_list[k] = t.MultiLevel(N, D, geff_list[k], epinput, wc, wa,rwa=False)
    systems_list[k].hamiltonian(suppress=True)
    systems_energies_list[k] = np.array(systems_list[k].H.eigenenergies())
    systems_energies_list[k] = systems_energies_list[k] - systems_energies_list[k].min()

###color stuff
systems_states_list = np.empty([geff_list_num], dtype = object)
for k in range(geff_list_num):
    systems_states_list[k] = np.array(systems_list[k].H.eigenstates()[1])
state_list = np.empty([len(systems_states_list[0])],dtype=object) #energy levels specifically!!
for n in range(len(state_list)): #the length is the same as N*D because hamiltonian diagonalisation is the amount of energy levels
    state_list[n] = [systems_states_list[k][n] for k in range(len(geff_list))]
for n in range(len(state_list)): #the length is the same as N*D because hamiltonian diagonalisation is the amount of energy levels
    for m in range(len(state_list[0])):
        state_list[n][m] = state_list[n][m]*state_list[n][m].dag()
darkness_list = np.empty([N*D,len(geff_list)], dtype = object)
for n in range(len(darkness_list)):
    for m in range(len(darkness_list[0])):
                   darkness_list[n][m] = systems_list[m].darkstate_proportion_external(state_list[n][m])
###
energy_list = np.empty([len(systems_energies_list[0])],dtype=object) #energy levels specifically!!
for n in range(len(energy_list)): #the length is the same as N*D because hamiltonian diagonalisation is the amount of energy levels
    energy_list[n] = [systems_energies_list[k][n] for k in range(len(geff_list))]



fig, ax = plt.subplots()
plt.xlim(geff_list_min,geff_list_max)
plt.ylim(-0.5, 3.5)

#original stuff 
# for n in range(30):#plotting
#     ELevelLines, = ax.plot(geff_list,energy_list[n], color = 'black', linestyle = '-', label = 'energies') #no rescaling?
        
#just testing cmap... you would need to do a scatter and plot points individually.
cmap = matplotlib.colormaps['copper']
for n in range(30):#plotting
    for m in range(len(geff_list)):
        ELevelLines = ax.scatter(geff_list_iterator[m],energy_list[n][m], label = 'energies', marker = '.', linewidths=0.5, c = cmap(1-darkness_list[n][m])) #no rescaling?
        
ax.set_ylabel(r'$(E+g_{eff}^2)/\omega$')
ax.set_xlabel(r'$g_{eff}/\omega$')
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap),
             ax=ax, label="Superradiance")
plt.title(r'Energy Spectrum of MQRM for ' + r'$D = $' + str(D-1) + r', $\varepsilon = $' + str(round(ep,3)) + r'$\omega$')
