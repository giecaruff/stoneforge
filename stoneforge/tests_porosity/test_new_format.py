# %%
import sys
import os
from gevent import config
import numpy as np
import pytest
from stoneforge.tests_porosity.test_improved import sorted_values

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

    config_1 = {
    "rhob":(0.0,1.0),
    "rhom":(0.0,1.0),
    "rhof":(0.0,1.0),
    }

    density_values = sorted_values(config_1)

    @pytest.mark.parametrize(density_values[1], density_values[0])
    def test_density(rhob, rhom, rhof):
        p = porosity(rhob = rhob, rhom = rhom, rhof = rhof,
                        method = "density")
        assert p >= 0 and p <= 1

density_test = Density()
density_test.sorted_values()