#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 00:35:48 2023

@author: tibbles
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
geff_forops = 1
ep=0.2*geff_forops
wa = 1            # cavity and atom frequency
wc = 1