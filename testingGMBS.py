#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:17:51 2023

@author: tibbles
"""

import numpy as np
from qutip import *
qutip.settings.has_mkl = False
import matplotlib.pyplot as plt
import simulation as t
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
sys=t.GeneralBlochSiegert(2,3,1,1,1,1)
sys.hamiltonian()
print(sys.phi)
print(sys.U1@ sys.U1.T)
print(sys.U2)
print(sys.U)
print(sys.U*sys.Udag)

