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

sys=t.GeneralBlochSiegert(2,3,0,0.5,1,1)
sys.hamiltonian()
print(sys.H_ep)