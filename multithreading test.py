# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 04:10:15 2022

@author: Tib
"""

import numpy as np
import random as r

length = 1000

final_array = np.empty([length],dtype=np.float64)
for k in range(length):
    final_array[k] = r.random()

print(final_array)