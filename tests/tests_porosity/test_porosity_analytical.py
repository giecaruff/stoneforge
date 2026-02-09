# %%
import pytest
import numpy as np

if __package__:
    from ..petrophysics.porosity import porosity
else:
    from stoneforge.petrophysics import porosity

# -------------------------------------------------------------------------------------------------------------- #
# test functions

def test_density_porosity_scalar():
    rhob = 2.35
    rhom = 2.65
    rhof = 1.10

    expected = (rhom - rhob) / (rhom - rhof)
    result = porosity.density_porosity(rhob=rhob, rhom=rhom, rhof=rhof)
    result2 = porosity.porosity(rhob=rhob, rhom=rhom, rhof=rhof, method='density')

    assert np.isclose(result, expected, rtol=1e-6)
    assert np.isclose(result2, expected, rtol=1e-6)
    
def test_effective_porosity_scalar():
    phi = 0.17
    vsh = 0.23
    phi_sh = 0.05

    expected = phi - (vsh * phi_sh)
    result = porosity.effective_porosity(phi=phi, vsh=vsh, phi_sh=phi_sh)
    result2 = porosity.porosity(phi=phi, vsh=vsh, phi_sh=phi_sh, method='effective')

    assert np.isclose(result, expected, rtol=1e-6)
    assert np.isclose(result2, expected, rtol=1e-6)
    
def test_neutron_porosity_scalar():
    nphi = 0.25
    vsh = 0.30
    phi_sh = 0.04

    expected = nphi - (vsh * phi_sh)
    result = porosity.neutron_porosity(nphi=nphi, vsh=vsh, phish=phi_sh)
    result2 = porosity.porosity(nphi=nphi, vsh=vsh, phish=phi_sh, method='neutron')

    assert np.isclose(result, expected, rtol=1e-6)
    assert np.isclose(result2, expected, rtol=1e-6)
    
def test_neutron_density_porosity_squared_scalar():
    nphi = 0.22
    phid = 0.18

    expected = np.sqrt((nphi**2 + phid**2) / 2)
    result = porosity.neutron_density_porosity(phin=nphi, phid=phid, squared=True)
    result2 = porosity.porosity(phin=nphi, phid=phid, squared=True, method='neutron-density')

    assert np.isclose(result, expected, rtol=1e-6)
    assert np.isclose(result2, expected, rtol=1e-6)
    
def test_neutron_density_porosity_non_squared_scalar():
    nphi = 0.22
    phid = 0.18

    expected = (nphi + phid) / 2
    result = porosity.neutron_density_porosity(phin=nphi, phid=phid, squared=False)
    result2 = porosity.porosity(phin=nphi, phid=phid, squared=False, method='neutron-density')
    assert np.isclose(result, expected, rtol=1e-6)
    assert np.isclose(result2, expected, rtol=1e-6)
    
def test_sonic_porosity_scalar():
    dt = 80.0
    dt_ma = 55.0
    dt_f = 185.0

    expected = (dt - dt_ma) / (dt_f - dt_ma)
    result = porosity.sonic_porosity(dt=dt, dtma=dt_ma, dtf=dt_f)
    result2 = porosity.porosity(dt=dt, dtma=dt_ma, dtf=dt_f, method='sonic')
    assert np.isclose(result, expected, rtol=1e-6)
    assert np.isclose(result2, expected, rtol=1e-6)
    
def test_gaymard_poupon_porosity_scalar():
    phid = 0.18
    phin = 0.22

    expected = np.sqrt((phid**2 + phin**2) / 2)
    result = porosity.gaymard_poupon_porosity(phid=phid, phin=phin)
    resilt2 = porosity.porosity(phid=phid, phin=phin, method='gaymard')
    assert np.isclose(result, expected, rtol=1e-6)
    assert np.isclose(resilt2, expected, rtol=1e-6)