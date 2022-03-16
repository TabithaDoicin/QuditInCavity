
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:13:10 2022

@author: Tib
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as anim
import vee_opsandbasis as v

N = 10               # number of cavity fock states
D = 4              #number of atomic states?
geff = 1
glist = np.linspace(geff/np.sqrt(D-1),geff/np.sqrt(D-1),D-1)
wc = wa = 0 # cavity and atom frequency
g  = 1    # coupling strength
kappa = g*0.05            # cavity dissipation rate
gamma = 0          # atom dissipation rate
gamma_d = 0.04*kappa
biggamma = 0.02*kappa
# Jaynes-Cummings Hamiltonian


a  = tensor(destroy(N), qeye(D))
sm = np.empty([D,D],dtype=object)

for n in range(1,D+1):
    for m in range(1,D+1):
        sm[n-1,m-1] = tensor(qeye(N),Qobj(v.vector(n,m,D)))
 
H = wc * a.dag() * a + wa * sum([sm[i,i] for i in range(D)]) + sum([glist[n-1]*(a.dag() * sm[0,n] + a * sm[n,0]) for n  in  range(1,D)])
print(H)
co_op_cavity_decay = [np.sqrt(kappa)*a]
co_op_radiative_decay = [np.sqrt(gamma)*sm[0,n] for n in range(1,D)]
co_op_dephasing = [np.sqrt(gamma_d)*sm[n,n] for n in  range(1,D)]
co_op_pumping = [np.sqrt(biggamma)*sm[n,0] for n in range(1,D)]
c_ops = co_op_cavity_decay+co_op_pumping+co_op_radiative_decay+co_op_dephasing


wlist = np.linspace(-np.pi *g + wc, np.pi *g + wc, 2000)
spec = 1*spectrum(H, wlist, c_ops, a.dag(), a)

fig, ax = plt.subplots()
ax.plot(wlist, np.log10(spec))