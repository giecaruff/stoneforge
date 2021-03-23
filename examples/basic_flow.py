# %%

"""
=======================================
Load and crop Dataset Exercise
=======================================

A tutorial exercise about loading well log datasets
"""

import las2 # local las2 read
import numpy as np
import matplotlib.pyplot as plt

lasfile = las2.read('../datasets/DP1.las')
DATA = {} # data information from DP1 welllog

for i in range(len(lasfile['curve'])):
    name = lasfile['curve'][i]['mnemonic']
    DATA[name] = lasfile['data'][i]
    print(lasfile['curve'][i])
    
# %%
