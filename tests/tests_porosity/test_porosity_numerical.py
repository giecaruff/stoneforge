# %%
import pytest
import numpy as np

if __package__:
    from ..petrophysics.porosity import porosity
else:
    from stoneforge.petrophysics import porosity

# -------------------------------------------------------------------------------------------------------------- #
# test functions

def test_density_porosity_vectorized_with_nan():
    rhob = np.array([2.30, 2.45, np.nan])
    rhom = 2.65
    rhof = 1.10

    expected = (rhom - rhob) / (rhom - rhof)
    result = porosity.density_porosity(rhob=rhob, rhom=rhom, rhof=rhof)

    assert isinstance(result, np.ndarray)
    assert np.isnan(result[2])
    assert np.allclose(result[:2], expected[:2], rtol=1e-6)
    
def test_effective_porosity_vectorized():
    phi = np.array([0.18, 0.22, 0.15])
    vsh = np.array([0.20, 0.35, 0.10])
    phi_sh = 0.05

    expected = phi - (vsh * phi_sh)
    result = porosity.effective_porosity(phi=phi, vsh=vsh, phi_sh=phi_sh)

    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected, rtol=1e-6)
    
def test_neutron_porosity_vectorized_bounds():
    nphi = np.array([0.30, 0.25, np.nan])
    vsh = np.array([0.20, 0.40, 0.10])
    phi_sh = 0.04

    result = porosity.neutron_porosity(nphi=nphi, vsh=vsh, phish=phi_sh)

    assert np.isnan(result[2])
    assert np.all(result[:2] >= 0.0)
    assert np.all(result[:2] <= 1.0)
    
def test_neutron_density_porosity_vectorized_modes():
    phin = np.array([0.20, 0.25, 0.30])
    phid = np.array([0.18, 0.22, 0.28])

    expected_squared = np.sqrt((phin**2 + phid**2) / 2)
    expected_linear = (phin + phid) / 2

    result_squared = porosity.neutron_density_porosity(
        phin=phin, phid=phid, squared=True
    )
    result_linear = porosity.neutron_density_porosity(
        phin=phin, phid=phid, squared=False
    )

    assert np.allclose(result_squared, expected_squared, rtol=1e-6)
    assert np.allclose(result_linear, expected_linear, rtol=1e-6)
    
def test_sonic_porosity_vectorized_monotonicity():
    dt = np.array([60.0, 80.0, 100.0])
    dtma = 55.0
    dtf = 185.0

    result = porosity.sonic_porosity(dt=dt, dtma=dtma, dtf=dtf)

    assert isinstance(result, np.ndarray)
    assert np.all(np.diff(result) > 0)
    
def test_gaymard_poupon_porosity_nan_behavior():
    phid = np.array([0.18, np.nan, 0.22])
    phin = np.array([0.20, 0.25, np.nan])

    result = porosity.gaymard_porosity(phid=phid, phin=phin)

    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert result[0] > 0.0