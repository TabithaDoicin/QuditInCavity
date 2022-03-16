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

N = 10                  # number of cavity fock states
wc = wa = 1 # cavity and atom frequency
g  = 1    # coupling strength
kappa = g*0.05            # cavity dissipation rate
gamma = 0           # atom dissipation rate
gamma_d = kappa
biggamma = 0.02*kappa
# Jaynes-Cummings Hamiltonian
a  = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))
H = wc * a.dag() * a + wa * sm.dag() * sm  + g * (a.dag() * sm + a * sm.dag())

co_op_cavity_decay = [np.sqrt(kappa)*a]
co_op_radiative_decay = [np.sqrt(gamma)*sm]
co_op_dephasing = [np.sqrt(gamma_d)*sm.dag()*sm]
co_op_pumping = [np.sqrt(biggamma)*sm.dag()]
c_ops = co_op_cavity_decay+co_op_radiative_decay+co_op_dephasing+co_op_pumping
print(H)
# calculate the correlation function using the mesolve solver, and then fft to
# obtain the spectrum. Here we need to make sure to evaluate the correlation
# function for a sufficient long time and sufficiently high sampling rate so
# that the discrete Fourier transform (FFT) captures all the features in the
# resulting spectrum.
wlist = np.linspace(-np.pi *g + wc, np.pi *g + wc, 200)
spec = 0.06*spectrum(H, wlist, c_ops, a.dag(), a)

fig, ax = plt.subplots()
ax.plot(wlist, np.log10(spec))