# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 00:27:53 2022

@author: Tib
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t
import scipy as sp
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{physics}')

N = 30             # number of cavity fock states
D = 3             #number of atomic states
geff_forops = 1
ep=0.25*geff_forops
wa = 1            # cavity and atom frequency
wc = 1

#system for extracting operators (they are the same for rwa and no rwa)
sys = t.MultiLevel(N, D, geff_forops, ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)
#looking at geff variation
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
systems_gndstate_MBS_list_toOrder = np.empty([geff_list_num], dtype = object)

for k in range(geff_list_num):
    systems_rwa_list[k] = t.MultiLevel(N, D, geff_list[k], ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)
    systems_rwa_list[k].hamiltonian(suppress=True)
    systems_gndstate_rwa_list[k] = systems_rwa_list[k].H.groundstate()[1]
    
    systems_no_rwa_list[k] = t.MultiLevel(N, D, geff_list[k], ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=False)
    systems_no_rwa_list[k].hamiltonian(suppress=True)
    systems_gndstate_no_rwa_list[k] = systems_no_rwa_list[k].H.groundstate()[1]
    
    systems_MBS_list[k] = t.GeneralBlochSiegert(N, D, geff_list[k], ep, wc, wa)
    systems_MBS_list[k].hamiltonian()
    systems_gndstate_MBS_list[k] = systems_MBS_list[k].Udag * systems_MBS_list[k].H.groundstate()[1] #basis changed
    systems_gndstate_MBS_list_toOrder[k] = systems_MBS_list[k].U_toOrder_dag * systems_MBS_list[k].H.groundstate()[1]

n_gnd_rwa = expect(sys.n_op, systems_gndstate_rwa_list)
n_gnd_no_rwa = expect(sys.n_op, systems_gndstate_no_rwa_list)
n_gnd_MBS = expect(sys.n_op, systems_gndstate_MBS_list)
n_gnd_MBS_toOrder = expect(sys.n_op, systems_gndstate_MBS_list_toOrder)
list_of_steps = np.empty([geff_list_num])

for k in range(min([N,geff_list_num])):
    if k==0:
        list_of_steps[k] = np.sqrt(wc**2-ep**2/4)
        pass
    else:
        list_of_steps[k] = np.sqrt(wc**2 * (2*k+1) + np.sqrt(wc**2 * (ep**2 + 4*k*wc + 4*k**2*wc**2)))#this is a specific solution of the quartic?
    
additionscaling = np.empty([len(geff_list)])
for k in range(len(geff_list)):
    additionscaling[k] = (geff_list[k])**2 

    
k = (D-2)*((wa+wc)/ep - 1/2)
n_gnd_analytical_MBS = [(g**2/(D-1)) * (D-2)**2/ep**2 * (sp.special.polygamma(1,k)-sp.special.polygamma(1,k+D-1)) for g in geff_list]


fig, ax = plt.subplots()
ax.set_ylabel(r'$\langle{a^\dagger a}\rangle$')
ax.set_xlabel(r'$g_{eff}$')
MJC, = ax.plot(geff_list, n_gnd_rwa, label='MJC', color='black')
MQRM, = ax.plot(geff_list, n_gnd_no_rwa-0*additionscaling, color='red', label='MQRM')
MBS_full, = ax.plot(geff_list, n_gnd_MBS, color='green', label='MBS_full')
#MBS_toOrder, = ax.plot(geff_list, n_gnd_MBS_toOrder, color='fuchsia', label='MBS_toOrder')
#MBS_toOrder_analytical, = ax.plot(geff_list, n_gnd_analytical_MBS,color = 'green', label='MBS_toOrder_analytical')
#ax.scatter(list_of_steps, np.linspace(0,0,len(list_of_steps)))
plt.title(r'Comparison of Groundstate Pop $\langle{a^\dagger a}\rangle$ Between Different Models for ' + 'D = ' + str(D-1))
plt.xlim(geff_list_min,geff_list_max)
ax.legend(handles=[MJC, MQRM, MBS_full])#, MBS_toOrder,MBS_toOrder_analytical])




