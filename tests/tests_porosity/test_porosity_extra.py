import pytest
import numpy as np
import warnings
from stoneforge.petrophysics import porosity

def test_density_porosity_warnings():
    # 1. rhom == rhof -> warning & returns np.nan
    with pytest.warns(UserWarning, match="This will result in a division by zero"):
        res = porosity.density_porosity(rhob=2.0, rhom=1.1, rhof=1.1)
        assert np.isnan(res)

    # 2. rhom <= rhof -> warning
    with pytest.warns(UserWarning, match="rhom must be greater than rhof and rhob"):
        porosity.density_porosity(rhob=2.0, rhom=1.0, rhof=1.2)

    # 3. rhob < rhof -> warning (rhom - rhob > rhom - rhof)
    with pytest.warns(UserWarning, match="rhob value is lower than rhof"):
        porosity.density_porosity(rhob=0.9, rhom=2.65, rhof=1.1)

def test_neutron_porosity_warnings():
    # 1. nphi < vsh * phish -> warning
    with pytest.warns(UserWarning, match="phin must be a positive value"):
        porosity.neutron_porosity(nphi=np.array([0.1]), vsh=np.array([0.5]), phish=0.48)

    # 2. phin > 1 -> warning
    with pytest.warns(UserWarning, match="phin must be a value between 0 and 1"):
        porosity.neutron_porosity(nphi=np.array([1.5]), vsh=np.array([0.0]), phish=0.1)

def test_neutron_correction_porosity():
    nphi = np.array([0.2, 0.3])
    lito = np.array([49, 30])
    res = porosity.neutron_correction_porosity(nphi, lito)
    # 49 -> sandstone -> nphi + 0.04 -> 0.24
    # 30 -> dolomite -> nphi - 0.06 -> 0.24
    assert np.allclose(res, [0.24, 0.24])

def test_neutron_density_porosity_warnings():
    # 1. (phid + phin / 2) > 1 -> warning
    with pytest.warns(UserWarning, match="phi must be a value between 0 and 1"):
        porosity.neutron_density_porosity(phid=0.9, phin=0.3, squared=False)
        
    # 2. (phid**2 + phin**2 / 2) > 1 -> warning
    with pytest.warns(UserWarning, match="phi must be a value between 0 and 1"):
        porosity.neutron_density_porosity(phid=0.9, phin=0.9, squared=True)

def test_sonic_porosity_warnings():
    # 1. dtf == dtma -> warning & nan
    with pytest.warns(UserWarning, match="This will result in a division by zero"):
        res = porosity.sonic_porosity(dt=100.0, dtma=50.0, dtf=50.0)
        assert np.isnan(res)

    # 2. dt <= dtma -> warning
    with pytest.warns(UserWarning, match="dt and dtf must be greater than dtma"):
        porosity.sonic_porosity(dt=40.0, dtma=50.0, dtf=180.0)

    # 3. dt > dtf -> warning
    with pytest.warns(UserWarning, match="dt value is greather than dtf"):
        porosity.sonic_porosity(dt=190.0, dtma=50.0, dtf=180.0)

def test_porosity_facade_type_error():
    # Missing required argument for density method
    with pytest.raises(TypeError) as excinfo:
        porosity.porosity(method="density", rhom=2.65)
    assert "Missing required argument" in str(excinfo.value)
