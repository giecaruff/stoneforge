# %%
import sys
import os


if __package__:
    from ..petrophysics.porosity import porosity
else:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from petrophysics.porosity import porosity


def test_density():
    assert round(porosity(rhob = 2.43, rhom = 2.65, rhof = 1.10,
                        method = "density"),2) == 0.14

def test_neutron():
    assert round(porosity(nphi = 0.18, vsh = 0.30, nphi_sh = 0.27,
                        method = "neutron"),2) == 0.10

def test_neutron_density():
    assert round(porosity(phid = 0.14, phin = 0.10, squared = False,
                        method = "neutron-density"),2) == 0.12

def test_sonic():
    assert round(porosity(dt = 80.0, dtma = 55.5, dtf = 185.0,
                        method = "sonic"),2) == 0.19

def test_gaymard():
    assert round(porosity(phid = 0.14, phin = 0.10,
                        method = "gaymard"),2) == 0.12


#TODO: pytest for the rest of porosity