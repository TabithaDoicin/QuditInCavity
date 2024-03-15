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
np.set_printoptions(precision=5)
g = 1
gefflist_normalised = 0.01*t.glist_generator(2,True)
sys_qrm = t.MultiLevel(3,3,gefflist_normalised,0.5,1,1,rwa=False)
