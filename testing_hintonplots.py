# -*- coding: utf-8 -*-U
"""
Created on Thu Jun  9 16:14:46 2022

@author: Tib
"""

#Steady state hinton animation plots

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import simulation as t
import matplotlib.animation as animation

N = 10         # number of cavity fock states
D = 4          #number of atomic states?
geff = 1
ep=0.5*geff
wa = 0 # cavity and atom frequency
wc = 0
kappa = 0.1*geff        # cavity dissipation rate
gamma = 0.1*kappa       # atom dissipation rate
gamma_d = 0.1*kappa
LAMBDA =0.5*kappa
omega = 2*geff


acc=100
system = t.MultiLevel(N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, LAMBDA, omega)
Hlist = system.hamiltonian(acc)
c_ops = system.collapse()

rho_ss_list = np.empty([len(Hlist)],dtype = object)

for k in range(len(Hlist)):
    rho_ss_list[k] = steadystate(Hlist[k],c_ops)



plasma = plt.cm.get_cmap(name='seismic', lut=None)
fig,axis = plt.subplots()


def animate(i):
    axis.clear()
    fig, ax = hinton(rho_ss_list[i],cmap = plasma,ax=axis)
    axis.set_title('%03d'%(i))

interval = 0.05#in seconds     
ani = animation.FuncAnimation(fig,animate,acc,interval=interval*1e+3,blit=False)
writergif = animation.PillowWriter(fps=20) 
f = r"c://Users/Anton/Desktop/animations/animation.gif"
ani.save(f, writer=writergif)