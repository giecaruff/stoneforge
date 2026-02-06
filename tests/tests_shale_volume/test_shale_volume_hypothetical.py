# %%
import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
import numpy as np

if __package__:
    from ..petrophysics.porosity import shale_volume
else:
    from stoneforge.petrophysics import shale_volume

# -------------------------------------------------------------------------------------------------------------- #
# test functions

@given(
    gr=st.floats(min_value=0.0, max_value=300.0),
    grmin=st.floats(min_value=0.0, max_value=50.0),
    grmax=st.floats(min_value=100.0, max_value=300.0),
)
def test_gammarayindex_properties(gr, grmin, grmax):
    assume(grmax > grmin)

    igr = shale_volume.gammarayindex(gr=gr, grmin=grmin, grmax=grmax)

    assert np.isfinite(igr)
    
@given(
    gr=st.floats(min_value=-50.0, max_value=350.0),
    grmin=st.floats(min_value=0.0, max_value=50.0),
    grmax=st.floats(min_value=100.0, max_value=300.0),
)
def test_gr_based_vshale_bounds(gr, grmin, grmax):
    assume(grmax > grmin)

    funcs = [
        shale_volume.vshale_linear,
        shale_volume.vshale_larionov_old,
        shale_volume.vshale_larionov,
        shale_volume.vshale_clavier,
        shale_volume.vshale_stieber,
    ]

    for func in funcs:
        vsh = func(gr=gr, grmin=grmin, grmax=grmax)
        assert np.isnan(vsh) or (0.0 <= vsh <= 1.0)
        
@given(
    gr1=st.floats(min_value=0.0, max_value=100.0),
    gr2=st.floats(min_value=100.0, max_value=250.0),
)
def test_gr_based_vshale_monotonicity(gr1, gr2):
    grmin = 0.0
    grmax = 150.0

    assume(gr1 < gr2)

    funcs = [
        shale_volume.vshale_linear,
        shale_volume.vshale_larionov_old,
        shale_volume.vshale_larionov,
        shale_volume.vshale_clavier,
        shale_volume.vshale_stieber,
    ]

    for func in funcs:
        vsh1 = func(gr=gr1, grmin=grmin, grmax=grmax)
        vsh2 = func(gr=gr2, grmin=grmin, grmax=grmax)

        assert vsh2 >= vsh1
        
@given(
    nphi=st.floats(min_value=0.0, max_value=0.6),
    rhob=st.floats(min_value=1.9, max_value=2.8),
)
def test_vshale_neutron_density_bounds(nphi, rhob):
    vsh = shale_volume.vshale_neu_den(
        nphi=nphi,
        rhob=rhob,
        clean_n=-0.15,
        clean_d=2.65,
        fluid_n=1.00,
        fluid_d=1.10,
        clay_n=0.47,
        clay_d=2.71,
    )

    assert np.isnan(vsh) or (0.0 <= vsh <= 1.0)
    
@given(
    phit=st.floats(min_value=0.05, max_value=0.4),
    phie=st.floats(min_value=0.0, max_value=0.4),
)
def test_vshale_nrm_physical_constraints(phit, phie):
    assume(phit > 0.0)
    assume(phie <= phit)

    vsh = shale_volume.vshale_nrm(phit=phit, phie=phie)

    assert 0.0 <= vsh <= 1.0
    
@given(
    grmin=st.floats(min_value=0.0, max_value=50.0),
    grmax=st.floats(min_value=100.0, max_value=300.0),
)
def test_vshale_linear_end_members(grmin, grmax):
    assume(grmax > grmin)

    vsh_min = shale_volume.vshale_linear(gr=grmin, grmin=grmin, grmax=grmax)
    vsh_max = shale_volume.vshale_linear(gr=grmax, grmin=grmin, grmax=grmax)

    assert np.isclose(vsh_min, 0.0)
    assert np.isclose(vsh_max, 1.0)

