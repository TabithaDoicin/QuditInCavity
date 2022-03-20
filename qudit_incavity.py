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
import operator_basis as v

N = 10               # number of cavity fock states
D = 7           #number of atomic states?
geff = 1
ep=1*geff
glist = np.linspace(geff/np.sqrt(D-1),geff/np.sqrt(D-1),D-1)
delta = np.linspace(-ep/2,ep/2,D-1)
wa = 0 # cavity and atom frequency
wc = 0
g  = 1    # coupling strength
kappa = g*0.05            # cavity dissipation rate
gamma = 0          # atom dissipation rate
gamma_d = 0.01*kappa
biggamma =0.05*kappa
# Jaynes-Cummings Hamiltonian


a  = tensor(destroy(N), qeye(D))
vec = np.empty([D,D],dtype=object)
vectorsmat = v.vector2(D) #new
# for n in range(D):
#     for m in range(D):
#         vec[n,m] = tensor(qeye(N),Qobj(v.vector(n+1,m+1,D)))
 
for n in range(D):
    for m in range(D):
        vec[n,m] = tensor(qeye(N),vectorsmat[n,m])
        
H = wc * a.dag() * a + sum([(wa + delta[i-1])*vec[i,i] for i in range(1,D)]) + sum([glist[n-1]*(a.dag() * vec[0,n] + a * vec[n,0]) for n  in  range(1,D)])
print(H)
co_op_cavity_decay = [np.sqrt(kappa)*a]
co_op_radiative_decay = [np.sqrt(gamma)*vec[0,n] for n in range(1,D)]
co_op_dephasing = [np.sqrt(gamma_d)*vec[n,n] for n in  range(1,D)]
co_op_pumping = [np.sqrt(biggamma)*vec[n,0] for n in range(1,D)]
c_ops = co_op_cavity_decay+co_op_pumping+co_op_radiative_decay+co_op_dephasing


wlist = np.linspace(-1*np.pi *g + wc, 1*np.pi *g + wc, 2000)
spec = 1*spectrum(H, wlist, c_ops, a.dag(), a)

fig, ax = plt.subplots()
ax.plot(wlist, np.log10(spec))