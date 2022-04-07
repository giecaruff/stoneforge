# %%
#this test format will be deprecated

import sys
import os
import numpy as np
import pytest

if __package__:
    from ..petrophysics.porosity import porosity
else:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from petrophysics.porosity import porosity


np.random.seed(99)

rhob = np.random.uniform(low=1.0, high=3.0, size=15)
rhom = np.random.uniform(low=1.0, high=2.5, size=15)
rhof = np.random.uniform(low=0.0, high=1.10, size=15)
density_values = np.array([rhob, rhom, rhof]).T

@pytest.mark.parametrize("rhob, rhom, rhof", density_values)
def test_density(rhob, rhom, rhof):
    p = porosity(rhob = rhob, rhom = rhom, rhof = rhof,
                        method = "density")
    assert p >= 0 and p <= 1


nphi = np.random.uniform(low=0.0, high=1.0, size=15)
vsh = np.random.uniform(low=0.0, high=1.0, size=15)
nphi_sh = np.random.uniform(low=0.0, high=1.0, size=15)
neutron_values = np.array([nphi, vsh, nphi_sh]).T

@pytest.mark.parametrize("nphi, vsh, nphi_sh", neutron_values)
def test_neutron(nphi, vsh, nphi_sh):
    p = porosity(nphi = nphi, vsh = vsh, nphi_sh = nphi_sh,
                        method = "neutron")
    assert p >= 0 and p <= 1


phid = np.random.uniform(low=0.0, high=1.0, size=15)
phin = np.random.uniform(low=0.0, high=1.0, size=15)
not_squared_neutron_density_values = np.array([phid, phin]).T

@pytest.mark.parametrize("phid, phin", not_squared_neutron_density_values)
def test_neutron_density(phid, phin):
    p = porosity(phid = phid, phin = phin, squared = False,
                        method = "neutron-density")
    assert p >= 0 and p <= 1


phid = np.random.uniform(low=0.0, high=1.0, size=15)
phin = np.random.uniform(low=0.0, high=1.0, size=15)
squared_neutron_density_values = np.array([phid, phin]).T

@pytest.mark.parametrize("phid, phin", squared_neutron_density_values)
def test_neutron_density_squared(phid, phin):
    p = porosity(phid = phid, phin = phin, squared = True,
                        method = "neutron-density")
    assert p >= 0 and p <= 1


dt = np.random.uniform(low=50.0, high=100.0, size=15)
dtma = np.random.uniform(low=10.0, high=100.0, size=15)
dtf = np.random.uniform(low=150.0, high=300.0, size=15)
sonic_values = np.array([dt, dtma, dtf]).T

@pytest.mark.parametrize("dt, dtma, dtf", sonic_values)
def test_sonic(dt, dtma, dtf):
    p = porosity(dt = dt, dtma = dtma, dtf = dtf,
                        method = "sonic")
    assert p >= 0 and p <= 1


phid = np.random.uniform(low=0.0, high=1.0, size=15)
phin = np.random.uniform(low=0.0, high=1.0, size=15)
gaymard_values = np.array([phid, phin]).T

@pytest.mark.parametrize("phid, phin", gaymard_values)
def test_gaymard(phid, phin):
    p = porosity(phid = phid, phin = phin,
                        method = "gaymard")
    assert p >= 0 and p <= 1


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
