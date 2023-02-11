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

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{physics}')

N = 50             # number of cavity fock states
D = 10             #number of atomic states
geff = 1
ep=0.2*geff
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
systems_gndstate_rwa_list = np.empty([geff_list_num], dtype = object)

systems_no_rwa_list = np.empty([geff_list_num], dtype = object)
systems_gndstate_no_rwa_list = np.empty([geff_list_num], dtype = object)

for k in range(geff_list_num):
    systems_rwa_list[k] = t.MultiLevel(N, D, geff_list[k], ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)
    systems_rwa_list[k].hamiltonian(suppress=True)
    systems_gndstate_rwa_list[k] = systems_rwa_list[k].H.groundstate()[1]
    
    systems_no_rwa_list[k] = t.MultiLevel(N, D, geff_list[k], ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=False)
    systems_no_rwa_list[k].hamiltonian(suppress=True)
    systems_gndstate_no_rwa_list[k] = systems_no_rwa_list[k].H.groundstate()[1]

n_gnd_rwa = expect(sys.n_op, systems_gndstate_rwa_list)
n_gnd_no_rwa = expect(sys.n_op, systems_gndstate_no_rwa_list)

additionscaling = np.empty([len(geff_list)])
for k in range(len(geff_list)):
    additionscaling[k] = (geff_list[k])**2 
    
fig, ax = plt.subplots()
ax.set_ylabel(r'$\langle{a^\dagger a}\rangle$')
ax.set_xlabel(r'$g_{eff}$')
ax.plot(geff_list, n_gnd_rwa)
ax.plot(geff_list, n_gnd_no_rwa-0*additionscaling)