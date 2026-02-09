# %%
from hypothesis import given, strategies as st, assume
import pytest
import numpy as np

if __package__:
    from ..petrophysics.porosity import porosity
else:
    from stoneforge.petrophysics import porosity

# -------------------------------------------------------------------------------------------------------------- #
# test functions

@given(
    rhob=st.floats(min_value=1.9, max_value=2.8),
    rhom=st.floats(min_value=2.6, max_value=2.9),
    rhof=st.floats(min_value=0.9, max_value=1.2),
)
def test_density_porosity_physical_bounds(rhob, rhom, rhof):
    assume(rhom > rhof)
    assume(rhob <= rhom)

    phi = porosity.density_porosity(rhob=rhob, rhom=rhom, rhof=rhof)

    assert np.isnan(phi) or (0.0 <= phi <= 1.0)
    
    
@given(
    rhob1=st.floats(min_value=2.0, max_value=2.4),
    rhob2=st.floats(min_value=2.4, max_value=2.8),
)
def test_density_porosity_monotonic(rhob1, rhob2):
    rhom = 2.65
    rhof = 1.10

    assume(rhob1 < rhob2)

    phi1 = porosity.density_porosity(rhob=rhob1, rhom=rhom, rhof=rhof)
    phi2 = porosity.density_porosity(rhob=rhob2, rhom=rhom, rhof=rhof)

    assert phi1 >= phi2
    

@given(
    phi=st.floats(min_value=0.05, max_value=0.35),
    vsh=st.floats(min_value=0.0, max_value=0.6),
    phi_sh=st.floats(min_value=0.0, max_value=0.1),
)
def test_effective_porosity_constraints(phi, vsh, phi_sh):
    phi_e = porosity.effective_porosity(phi=phi, vsh=vsh, phi_sh=phi_sh)

    assert phi_e <= phi
    
    
@given(
    nphi=st.floats(min_value=0.05, max_value=0.45),
    vsh=st.floats(min_value=0.0, max_value=0.6),
    phi_sh=st.floats(min_value=0.0, max_value=0.1),
)
def test_neutron_porosity_physical_bounds(nphi, vsh, phi_sh):
    phi = porosity.neutron_porosity(nphi=nphi, vsh=vsh, phish=phi_sh)

    assert np.isnan(phi) or (0.0 <= phi <= 1.0)
    
    
@given(
    phin=st.floats(min_value=0.05, max_value=0.45),
    phid=st.floats(min_value=0.05, max_value=0.45),
)
def test_neutron_density_rms_ge_mean(phin, phid):
    rms = porosity.neutron_density_porosity(phin=phin, phid=phid, squared=True)
    mean = porosity.neutron_density_porosity(phin=phin, phid=phid, squared=False)

    assert rms >= mean
    
    
@given(
    dt=st.floats(min_value=55.0, max_value=185.0),
    dtma=st.floats(min_value=50.0, max_value=70.0),
    dtf=st.floats(min_value=170.0, max_value=200.0),
)
def test_sonic_porosity_physical_behavior(dt, dtma, dtf):
    assume(dtf > dtma)
    assume(dt >= dtma)

    phi = porosity.sonic_porosity(dt=dt, dtma=dtma, dtf=dtf)

    assert np.isnan(phi) or (0.0 <= phi <= 1.0)
    
    
@given(
    phid=st.floats(min_value=0.05, max_value=0.45),
    phin=st.floats(min_value=0.05, max_value=0.45),
)
def test_gaymard_poupon_symmetry(phid, phin):
    gp1 = porosity.gaymard_poupon_porosity(phid=phid, phin=phin)
    gp2 = porosity.gaymard_poupon_porosity(phid=phin, phin=phid)

    assert np.isclose(gp1, gp2, rtol=1e-6)