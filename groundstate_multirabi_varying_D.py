# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:27:17 2023

@author: Tib
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{physics}')

N = 40             # number of cavity fock states
#D = 2             #number of atomic states
geff = 0.2
ep=0.5
wa = 1            # cavity and atom frequency
wc = 1


#looking at D variation
D_list_min = 2
D_list_max = 20
D_list = range(D_list_min, D_list_max+1)

systems_rwa_list = np.empty([len(D_list)], dtype = object)
systems_gndstate_rwa_list = np.empty([len(D_list)], dtype = object)

systems_no_rwa_list = np.empty([len(D_list)], dtype = object)
systems_gndstate_no_rwa_list = np.empty([len(D_list)], dtype = object)

for k in range(len(D_list)):
    systems_rwa_list[k] = t.MultiLevel(N, D_list[k], geff, ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=True)
    systems_rwa_list[k].hamiltonian(suppress=True)
    systems_gndstate_rwa_list[k] = systems_rwa_list[k].H.groundstate()[1]
    
    systems_no_rwa_list[k] = t.MultiLevel(N, D_list[k], geff, ep, wc, wa, 0, 0, 0, 0, 0, 0, 0, rwa=False)
    systems_no_rwa_list[k].hamiltonian(suppress=True)
    systems_gndstate_no_rwa_list[k] = systems_no_rwa_list[k].H.groundstate()[1]
    print(k+D_list_min)

n_gnd_rwa = np.empty([len(D_list)])
n_gnd_no_rwa = np.empty([len(D_list)])

for k in range(len(D_list)):
    n_gnd_rwa[k] = expect(systems_rwa_list[k].n_op, systems_gndstate_rwa_list[k])
    n_gnd_no_rwa[k] = expect(systems_no_rwa_list[k].n_op, systems_gndstate_no_rwa_list[k])

fig, ax = plt.subplots()
ax.set_ylabel(r'$\langle{a^\dagger a}\rangle$')
ax.set_xlabel(r'$D$')
ax.set_title("Gnd_state pops for rwa(blue) and no_rwa(orange) varying D, with geff=" + str(geff) + ", ep=" + str(ep) + ".")
ax.plot(D_list, n_gnd_rwa)
ax.plot(D_list, n_gnd_no_rwa)
