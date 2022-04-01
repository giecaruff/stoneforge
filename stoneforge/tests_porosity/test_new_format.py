# %%
import sys
import os
from typing_extensions import Self
import numpy as np
import pytest

if __package__:
    from ..petrophysics.porosity import porosity
else:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from petrophysics.porosity import porosity

# ---------------------------------------------------------- #
# function

class Tests:

    def sorted_values (configuration, size = 15, seed = 99):

        np.random.seed(seed)

        # transform ["a","b","c"] into "a,b,c"
        list_names = list(configuration.keys())
        values_names = ','.join(list_names)

        property_values = []
        for k in configuration:
            property = np.random.uniform(low = configuration[k][0], high = configuration[k][1], size = size)
            property_values.append(property)

        property_values = np.array(property_values).T

        return property_values,values_names

    config_1 = {
    "rhob":(0.0,1.0),
    "rhom":(0.0,1.0),
    "rhof":(0.0,1.0),
    }

    config_2 = {
    "nphi":(0.0,1.0),
    "vsh":(0.0,1.0),
    "nphi_sh":(0.0,1.0),
    }

    config_3 = {
    "phid":(0.0,1.0),
    "phin":(0.0,1.0),
    }

    config_4 = {
    "phid":(0.0,1.0),
    "phin":(0.0,1.0),
    }

    config_5 = {
    "dt":(50.0,100.0),
    "dtma":(10.0,100.0),
    "dtf":(150.0,300.0),
    }

    config_6 = {
    "phid":(0.0,1.0),
    "phin":(0.0,1.0),
    }

# ---------------------------------------------------------- #

class Density(Tests):

    def __init__ (self):
        
        self.config_1 = {
        "rhob":(0.0,1.0),
        "rhom":(0.0,1.0),
        "rhof":(0.0,1.0),
        }

        self.density_values = self.sorted_values(self.config_1)

    def values(self):

        @pytest.mark.parametrize(self.density_values[1], self.density_values[0])
        def test_density(rhob, rhom, rhof):
            p = porosity(rhob = rhob, rhom = rhom, rhof = rhof,
                        method = "density")
            assert p >= 0 and p <= 1

density_test = Density()
density_test.values()
print(density_test.density_values)
# ---------------------------------------------------------- #

""" class Neutron(Tests):

    config_2 = {
    "rhob":(0.0,1.0),
    "rhom":(0.0,1.0),
    "rhof":(0.0,1.0),
    }

    neutron_values = sorted_values(config_2)

    @pytest.mark.parametrize(neutron_values[1], neutron_values[0])
    def test_neutron(nphi, vsh, nphi_sh):
        p = porosity(nphi = nphi, vsh = vsh, nphi_sh = nphi_sh,
                        method = "neutron")
        assert p >= 0 and p <= 1 """

# ---------------------------------------------------------- #