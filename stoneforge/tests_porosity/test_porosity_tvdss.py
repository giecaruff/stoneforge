# %%
import sys
import os
#import parameters_p
import numpy as np
from parameters_p import Parameters
import matplotlib.pyplot as plt

if __package__:
    from ..petrophysics.porosity import porosity
else:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from petrophysics.shale_volume import vshale_neu_den

# -------------------------------------------------------------------------------------------------------------- #
# test functions


# %%

neutron = np.array([0.20, 0.25, 0.30])
density = np.array([2.10, 2.15, 2.20])


# %%

vshale_neu_den(neutron, density)

_cl1 = [-0.15, 2.65]
_cl2 = [1.00, 1.10]
_clay = [0.47, 2.71]

plt.plot(_cl1[0],_cl1[1], '.r')
plt.plot(_cl2[0],_cl2[1], '.b')
plt.plot(_clay[0], _clay[1], '.g')

plt.plot(_cl1,_cl2, '-k')

plt.grid()
plt.show()


# %%
