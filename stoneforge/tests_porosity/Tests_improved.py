# %%
import sys
import os
import numpy as np
import pytest

if __package__:
    from ..petrophysics.porosity import porosity
else:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from petrophysics.porosity import porosity

# ---------------------------------------------------------- #
# function

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

# ---------------------------------------------------------- #
# where all modifications will be applied

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


config_2 = {
    "nphi":(0.0,1.0),
    "vsh":(0.0,1.0),
    "nphi_sh":(0.0,1.0),
}

neutron_values = sorted_values(config_2)

@pytest.mark.parametrize(neutron_values[1], neutron_values[0])
def test_neutron(nphi, vsh, nphi_sh):
    p = porosity(nphi = nphi, vsh = vsh, nphi_sh = nphi_sh,
                        method = "neutron")
    assert p >= 0 and p <= 1


config_3 = {
    "phid":(0.0,1.0),
    "phin":(0.0,1.0),
}

not_squared_neutron_density_values = sorted_values(config_3)

@pytest.mark.parametrize(not_squared_neutron_density_values[1], not_squared_neutron_density_values[0])
def test_neutron_density(phid, phin):
    p = porosity(phid = phid, phin = phin, squared = False,
                        method = "neutron-density")
    assert p >= 0 and p <= 1


config_4 = {
    "phid":(0.0,1.0),
    "phin":(0.0,1.0),
}

squared_neutron_density_values = sorted_values(config_4)

@pytest.mark.parametrize(squared_neutron_density_values[1], squared_neutron_density_values[0])
def test_neutron_density(phid, phin):
    p = porosity(phid = phid, phin = phin, squared = True,
                        method = "neutron-density")
    assert p >= 0 and p <= 1


config_5 = {
    "dt":(50.0,100.0),
    "dtma":(10.0,100.0),
    "dtf":(150.0,300.0),
}

sonic_values = sorted_values(config_5)

@pytest.mark.parametrize(sonic_values[1], sonic_values[0])
def test_sonic(dt, dtma, dtf):
    p = porosity(dt = dt, dtma = dtma, dtf = dtf,
                        method = "sonic")
    assert p >= 0 and p <= 1


config_6 = {
    "phid":(0.0,1.0),
    "phin":(0.0,1.0),
}

gaymard_values = sorted_values(config_6)

@pytest.mark.parametrize(gaymard_values[1], gaymard_values[0])
def test_gaymard(phid, phin):
    p = porosity(phid = phid, phin = phin,
                        method = "gaymard")
    assert p >= 0 and p <= 1
# ---------------------------------------------------------- #


unique_density_value = []
rhob = 2.60
rhom = 1.10
rhof = 1.10
unique_density_value.append((rhob, rhom, rhof))

@pytest.mark.parametrize("rhob, rhom, rhof", unique_density_value)
def test_unique_density(rhob, rhom, rhof):
    p = porosity(rhob = 2.60, rhom = 1.10, rhof = 1.10,
                        method = "density")
    assert p >= 0 and p <= 1


unique_sonic_value = []
dt = 75.0
dtma = 100.0
dtf = 100.0
unique_sonic_value.append((dt, dtma, dtf))

@pytest.mark.parametrize("dt, dtma, dtf", unique_sonic_value)
def test_unique_sonic(dt, dtma, dtf):
    p = porosity(dt = 75.0, dtma = 100.0, dtf = 100.0,
                        method = "sonic")
    assert p >= 0 and p <= 1

#TODO: pytest for the rest of porosity