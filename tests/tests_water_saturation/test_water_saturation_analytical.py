# %%
import pytest
import numpy as np

if __package__:
    from ..petrophysics.porosity import water_saturation
else:
    from stoneforge.petrophysics import water_saturation

# -------------------------------------------------------------------------------------------------------------- #
# test functions

def test_archie_scalar():
    rt = 200.0
    phi = 0.25
    rw = 0.02
    a = 1.0
    m = 2.0
    n = 2.0
    
    expected = ((a*rw) / (phi**m * rt))**(1/n)
    expected = np.clip(expected, 0.0, 1.0)
    result = water_saturation.archie(rt=rt, phi=phi, rw=rw, a=a, m=m, n=n)
    assert np.isclose(result, expected, rtol=1e-6)

def test_simandoux_scalar():
    rt = 200.0
    phi = 0.25
    rw = 0.02
    a = 1.0
    m = 2.0
    n = 2.0
    vsh = 0.1
    rsh = 4.0
    
    C = (1 - vsh) * a * rw / phi**m
    D = C * vsh / (2*rsh)
    E = C / rt
    expected = ((D**2 + E)**0.5 - D)**(2/n)
    expected = np.clip(expected, 0.0, 1.0)
    result = water_saturation.simandoux(rt=rt, phi=phi, rw=rw, a=a, m=m, n=n, vsh=vsh, rsh=rsh)
    assert np.isclose(result, expected, rtol=1e-6)
    
def test_indonesia_scalar():
    rt = 200.0
    phi = 0.25
    rw = 0.02
    a = 1.0
    m = 2.0
    n = 2.0
    vsh = 0.1
    rsh = 4.0
    
    C = (1./rt) ** 0.5
    D = 1 - 0.5*vsh
    E = (vsh**D)/(rsh**0.5)
    F = ((phi**m)/(a*rw))**0.5
    expected = (C/(E+F))**(1/n)
    expected = np.clip(expected, 0.0, 1.0)
    result = water_saturation.indonesia(rt=rt, phi=phi, rw=rw, a=a, m=m, n=n, vsh=vsh, rsh=rsh)
    assert np.isclose(result, expected, rtol=1e-6)
    
def test_fertl_scalar():
    rt = 200.0
    phi = 0.25
    rw = 0.02
    a = 1.0
    m = 2.0
    vsh = 0.1
    alpha = 0.5
    
    expected = phi**(-m/2) * ((a*rw/rt + (alpha*vsh/2)**2)**0.5 - (alpha*vsh/2))
    expected = np.clip(expected, 0.0, 1.0)
    result = water_saturation.fertl(rt=rt, phi=phi, rw=rw, a=a, m=m, vsh=vsh, alpha=alpha)
    assert np.isclose(result, expected, rtol=1e-6)