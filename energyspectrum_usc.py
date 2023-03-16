# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:28:32 2022

@author: Tib
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

N = 50             # number of cavity fock states #needs to be really high to properly classify eigenenergies
D = 5             #number of atomic states
geff = 1
ep=0*geff
wa = 1            # cavity and atom frequency
wc = 1

#system for extracting operators (they are the same for rwa and no rwa)
sys = t.MultiLevel(N, D, geff, ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)

#looking at geff variation
geff_list_min = 0
geff_list_max = 5
geff_list_num = 200

geff_list = np.linspace(geff_list_min, geff_list_max, geff_list_num)

systems_rwa_list = np.empty([geff_list_num], dtype = object)
systems_energies_rwa_list = np.empty([geff_list_num], dtype = object)

systems_no_rwa_list = np.empty([geff_list_num], dtype = object)
systems_energies_no_rwa_list = np.empty([geff_list_num], dtype = object)

systems_MBS_list = np.empty([geff_list_num], dtype = object)
systems_energies_MBS_list = np.empty([geff_list_num], dtype = object)

for k in range(geff_list_num):
    systems_rwa_list[k] = t.MultiLevel(N, D, geff_list[k], ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)
    systems_rwa_list[k].hamiltonian(suppress=True)
    systems_energies_rwa_list[k] = np.array(systems_rwa_list[k].H.eigenenergies())
    
    systems_no_rwa_list[k] = t.MultiLevel(N, D, geff_list[k], ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=False)
    systems_no_rwa_list[k].hamiltonian(suppress=True)
    systems_energies_no_rwa_list[k] = np.array(systems_no_rwa_list[k].H.eigenenergies())
    
    systems_MBS_list[k] = t.DegenBlochSiegert(N, D, geff_list[k], wc, wa)
    systems_MBS_list[k].hamiltonian()
    systems_energies_MBS_list[k] = np.array(systems_MBS_list[k].H.eigenenergies())

energy_rwa_list = np.empty([len(systems_energies_rwa_list[0])],dtype=object) #energy levels specifically!!
energy_no_rwa_list = np.empty([len(systems_energies_no_rwa_list[0])],dtype=object) #energy levels specifically!!
energy_MBS_list = np.empty([len(systems_energies_MBS_list[0])],dtype=object) #energy levels specifically!!

for n in range(len(energy_rwa_list)): #the length is the same as N*D because hamiltonian diagonalisation is the amount of energy levels
    energy_rwa_list[n] = [systems_energies_rwa_list[k][n] for k in range(len(geff_list))]
    energy_no_rwa_list[n] =  [systems_energies_no_rwa_list[k][n] for k in range(len(geff_list))]
    energy_MBS_list[n] =  [systems_energies_MBS_list[k][n] for k in range(len(geff_list))]
    
fig, ax = plt.subplots()
additionscaling = np.empty([len(geff_list)])
plt.ylim(-2, 5)

for k in range(len(geff_list)):
    additionscaling[k] = (geff_list[k])**2 
    
#ax.plot(geff_list,energy_rwa_list[0], color = 'black', linestyle = 'dotted')
#ax.plot(geff_list,energy_no_rwa_list[0]+0*additionscaling, color = 'red')

for n in range(len(energy_rwa_list)):#plotting
    #ax.plot(geff_list,energy_rwa_list[n], color = 'black', linestyle = 'dotted') #no rescaling?
    ax.plot(geff_list,energy_no_rwa_list[n]+1*additionscaling, color = 'red')
    ax.plot(geff_list,energy_MBS_list[n]+1*additionscaling, color = 'blue',linestyle = 'dotted')
    
ax.set_ylabel(r'E')
ax.set_xlabel(r'$g_{eff}$')
plt.title(r'Energy Spectrum of Mulilevel Q-Rabi model (red) and Mulilevel JC model (black)' + ', D=' + str(D))