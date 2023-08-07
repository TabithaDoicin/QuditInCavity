# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:28:32 2022

@author: Tib
"""

import numpy as np
from qutip import *
import scipy as sp
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

N = 40             # number of cavity fock states #needs to be really high to properly classify eigenenergies
D = 3          #number of atomic states
geff = 1
ep=0.25*geff
wa = 1            # cavity and atom frequency
wc = 1

#system for extracting operators (they are the same for rwa and no rwa)
sys = t.MultiLevel(N, D, geff, ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)

#looking at geff variation
geff_list_min = 0
geff_list_max = 3
geff_list_num = 200

geff_list = np.linspace(geff_list_min, geff_list_max, geff_list_num)

systems_rwa_list = np.empty([geff_list_num], dtype = object)
systems_energies_rwa_list = np.empty([geff_list_num], dtype = object)

systems_no_rwa_list = np.empty([geff_list_num], dtype = object)
systems_energies_no_rwa_list = np.empty([geff_list_num], dtype = object)

systems_GMBS_list = np.empty([geff_list_num], dtype = object)
systems_energies_GMBS_list = np.empty([geff_list_num], dtype = object)
systems_eigstates_GMBS_list = np.empty([geff_list_num], dtype = object)

systems_energies_GMBS_corrected_list = np.empty([geff_list_num,N*D], dtype = object) #<gmbs|H_mqrm|gmbs>

for k in range(geff_list_num):
    systems_rwa_list[k] = t.MultiLevel(N, D, geff_list[k], ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)
    systems_rwa_list[k].hamiltonian(suppress=True)
    systems_energies_rwa_list[k] = np.array(systems_rwa_list[k].H.eigenenergies())
    
    systems_no_rwa_list[k] = t.MultiLevel(N, D, geff_list[k], ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=False)
    systems_no_rwa_list[k].hamiltonian(suppress=True)
    systems_energies_no_rwa_list[k] = np.array(systems_no_rwa_list[k].H.eigenenergies())
    
    systems_GMBS_list[k] = t.GeneralBlochSiegert(N, D, geff_list[k], ep, wc, wa)
    systems_GMBS_list[k].hamiltonian()
    systems_energies_GMBS_list[k] = np.array(systems_GMBS_list[k].H.eigenenergies())
    
    systems_eigstates_GMBS_list[k] = np.array(systems_GMBS_list[k].Udag * systems_GMBS_list[k].H.eigenstates()[1])
    for j in range(N*D):
        boop = systems_eigstates_GMBS_list[k][j].dag()*systems_no_rwa_list[k].H*systems_eigstates_GMBS_list[k][j]
        systems_energies_GMBS_corrected_list[k][j] = np.real(boop[0][0][0])
        
energy_rwa_list = np.empty([len(systems_energies_rwa_list[0])],dtype=object) #energy levels specifically!!
energy_no_rwa_list = np.empty([len(systems_energies_no_rwa_list[0])],dtype=object) #energy levels specifically!!
energy_GMBS_list = np.empty([len(systems_energies_GMBS_list[0])],dtype=object) #energy levels specifically!!
energy_GMBS_corrected_list = np.empty([len(systems_energies_GMBS_corrected_list[0])],dtype=object)

for n in range(len(energy_rwa_list)): #the length is the same as N*D because hamiltonian diagonalisation is the amount of energy levels
    energy_rwa_list[n] = [systems_energies_rwa_list[k][n] for k in range(len(geff_list))]
    energy_no_rwa_list[n] =  [systems_energies_no_rwa_list[k][n] for k in range(len(geff_list))]
    energy_GMBS_list[n] =  [systems_energies_GMBS_list[k][n] for k in range(len(geff_list))]
    energy_GMBS_corrected_list[n] =  [systems_energies_GMBS_corrected_list[k][n] for k in range(len(geff_list))]
    
fig, ax = plt.subplots()
additionscaling = np.empty([len(geff_list)])
plt.ylim(-2, 5)

for k in range(len(geff_list)):
    additionscaling[k] = (geff_list[k])**2 
    
#ax.plot(geff_list,energy_rwa_list[0], color = 'black', linestyle = 'dotted')
#ax.plot(geff_list,energy_no_rwa_list[0]+0*additionscaling, color = 'red')

if D!=2 and ep!=0:
    k = (D-2)*((wa+wc)/ep - 1/2)
    E0 = [(-g**2/(D-1)) * (D-2)/ep * (sp.special.digamma(k+D-1)-sp.special.digamma(k)) for g in geff_list]
else:
    k=0
    E0 = [-g**2/(wa+wc) for g in geff_list]

for n in range(len(energy_rwa_list)):#plotting
    MJC_line, = ax.plot(geff_list,energy_rwa_list[n], color = 'black', linestyle = 'dotted', label='MJC') #no rescaling?
    MQRM_line, = ax.plot(geff_list,energy_no_rwa_list[n]+1*additionscaling, color = 'red', label='MQRM', linestyle = '-')
    #GMBS_line, = ax.plot(geff_list,energy_GMBS_list[n]+1*additionscaling, color = 'green',linestyle = '--', label='MBSM')
    #GMBS_gnd_analytical_line, = ax.plot(geff_list,E0+1*additionscaling, color = 'green', label = 'GND_GMBS')
    #GMBS_corrected_line, = ax.plot(geff_list,energy_GMBS_corrected_list[n]+1*additionscaling, color = 'fuchsia',linestyle = 'dotted', label='GMBS_corr')

ax.set_ylabel(r'$(E+g_{eff}^2)/\omega$')
ax.set_xlabel(r'$g_{eff}/\omega$')
plt.title(r'Comparison of Energy Spectrums for ' + r'$D = $' + str(D-1) + r', $\varepsilon = $' + str(round(ep,3)))
ax.legend(handles=[MJC_line, MQRM_line])#, GMBS_line])#,GMBS_gnd_analytical_line])#, GMBS_corrected_line])



