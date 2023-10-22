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
np.set_printoptions(precision=2)
g = 1
sys_qrm = t.MultiLevel(3,4,g,0.5,1,1,rwa=False)
sys_bsm =t.GeneralBlochSiegert(3,4,g,0.5,1,1)

Hqrm=sys_qrm.hamiltonian()
print(Hqrm)
Hbsm=sys_bsm.hamiltonian()
print(Hbsm)

Hqrm_to_bsm = sys_qrm.U * Hqrm * sys_qrm.Udag
Nqrm_to_bsm = sys_qrm.U * sys_qrm.n_op_tot * sys_qrm.Udag
print(Hqrm_to_bsm)
print(Nqrm_to_bsm)

### [H_bsm,N_qrm]
print(operators.commutator(Hbsm,sys_qrm.n_op_tot))
### [H_bsm,N_bsm] = [H_bsm, U N_qrm Udag]
print(operators.commutator(Hbsm,Nqrm_to_bsm))

print(operators.commutator(Hqrm,Nqrm_to_bsm))

print(operators.commutator(Hqrm,sys_qrm.n_op_tot))