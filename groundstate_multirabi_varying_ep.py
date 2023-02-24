# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:09:26 2023

@author: Tabitha
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{physics}')

N = 50             # number of cavity fock states
D = 3             #number of atomic states
geff = 1
ep_forops=0.2*geff
wa = 1            # cavity and atom frequency
wc = 1

#system for extracting operators (they are the same for rwa and no rwa)
sys = t.MultiLevel(N, D, geff, ep_forops, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)
#looking at geff variation
ep_list_min = 0*geff
ep_list_max = 5*geff
ep_list_num = 200

ep_list = np.linspace(ep_list_min, ep_list_max, ep_list_num)

systems_rwa_list = np.empty([ep_list_num], dtype = object)
systems_gndstate_rwa_list = np.empty([ep_list_num], dtype = object)

systems_no_rwa_list = np.empty([ep_list_num], dtype = object)
systems_gndstate_no_rwa_list = np.empty([ep_list_num], dtype = object)

for k in range(ep_list_num):
    systems_rwa_list[k] = t.MultiLevel(N, D, geff, ep_list[k], wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)
    systems_rwa_list[k].hamiltonian(suppress=True)
    systems_gndstate_rwa_list[k] = systems_rwa_list[k].H.groundstate()[1]
    
    systems_no_rwa_list[k] = t.MultiLevel(N, D, geff, ep_list[k], wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=False)
    systems_no_rwa_list[k].hamiltonian(suppress=True)
    systems_gndstate_no_rwa_list[k] = systems_no_rwa_list[k].H.groundstate()[1]

n_gnd_rwa = expect(sys.n_op, systems_gndstate_rwa_list)
n_gnd_no_rwa = expect(sys.n_op, systems_gndstate_no_rwa_list)
    
fig, ax = plt.subplots()
ax.set_ylabel(r'$\langle{a^\dagger a}\rangle$')
ax.set_xlabel(r'$\varepsilon$')
ax.plot(ep_list, n_gnd_rwa)
ax.plot(ep_list, n_gnd_no_rwa)