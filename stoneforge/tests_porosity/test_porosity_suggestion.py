# %%
import sys
import os
import pytest
from parameters import Parameters

if __package__:
    from ..petrophysics.porosity import porosity
else:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from petrophysics.porosity import porosity

# -------------------------------------------------------------------------------------------------------------- #
# test functions


density_values = Parameters.sorted_values(Parameters.config_density)

@pytest.mark.parametrize(density_values[1], density_values[0])
def test_density(rhob, rhom, rhof):
    p = porosity(rhob = rhob, rhom = rhom, rhof = rhof,
                        method = "density")
    assert p >= 0 and p <= 1

# -------------------------------------------------------------------------------------------------------------- #

neutron_values = Parameters.sorted_values(Parameters.config_neutron)

@pytest.mark.parametrize(neutron_values[1], neutron_values[0])
def test_neutron(nphi, vsh, nphi_sh):
    p = porosity(nphi = nphi, vsh = vsh, nphi_sh = nphi_sh,
                        method = "neutron")
    assert p >= 0 and p <= 1 

# -------------------------------------------------------------------------------------------------------------- #

not_squared_neutron_density_values = Parameters.sorted_values(Parameters.config_neutron_density)

@pytest.mark.parametrize(not_squared_neutron_density_values[1], not_squared_neutron_density_values[0])
def test_neutron_density_not_squared(phid, phin):
    p = porosity(phid = phid, phin = phin, squared = False,
                        method = "neutron-density")
    assert p >= 0 and p <= 1

# -------------------------------------------------------------------------------------------------------------- #

squared_neutron_density_values = Parameters.sorted_values(Parameters.config_neutron_density)

@pytest.mark.parametrize(squared_neutron_density_values[1], squared_neutron_density_values[0])
def test_neutron_density_squared(phid, phin):
    p = porosity(phid = phid, phin = phin, squared = True,
                        method = "neutron-density")
    assert p >= 0 and p <= 1

# -------------------------------------------------------------------------------------------------------------- #

sonic_values = Parameters.sorted_values(Parameters.config_sonic)

@pytest.mark.parametrize(sonic_values[1], sonic_values[0])
def test_sonic(dt, dtma, dtf):
    p = porosity(dt = dt, dtma = dtma, dtf = dtf,
                        method = "sonic")
    assert p >= 0 and p <= 1

# -------------------------------------------------------------------------------------------------------------- #

gaymard_values = Parameters.sorted_values(Parameters.config_gaymard)

@pytest.mark.parametrize(gaymard_values[1], gaymard_values[0])
def test_gaymard(phid, phin):
    p = porosity(phid = phid, phin = phin,
                        method = "gaymard")
    assert p >= 0 and p <= 1

# -------------------------------------------------------------------------------------------------------------- #

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

# -------------------------------------------------------------------------------------------------------------- #

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

# TODO: pytest for the rest of porosity