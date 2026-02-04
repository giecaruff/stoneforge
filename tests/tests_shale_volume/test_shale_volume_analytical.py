# %%
import pytest
import numpy as np

if __package__:
    from ..petrophysics.porosity import shale_volume
else:
    from stoneforge.petrophysics import shale_volume

# -------------------------------------------------------------------------------------------------------------- #
# test functions

def test_gammarayindex_shaliness_scalar():
    gr = 120.0
    grmin = 0.0
    grmax = 150.0

    expected = (gr - grmin) / (grmax - grmin)
    result = shale_volume.gammarayindex(gr=gr, grmin=grmin, grmax=grmax)
    assert np.isclose(result, expected, rtol=1e-6)
    
def test_vshale_linear_shaliness_scalar():
    gr = 120.0
    grmin = 0.0
    grmax = 150.0

    expected_igr = (gr - grmin) / (grmax - grmin)
    expected_vshale = np.clip(expected_igr, 0.0, 1.0)

    result = shale_volume.vshale_linear(gr=gr, grmin=grmin, grmax=grmax)
    assert np.isclose(result, expected_vshale, rtol=1e-6)

def test_vshale_larionov_old_shaliness_scalar():
    gr = 120.0
    grmin = 0.0
    grmax = 150.0

    expected_igr = (gr - grmin) / (grmax - grmin)
    expected_vshale = 0.33 * (2. ** (2. * expected_igr) - 1)
    expected_vshale = np.clip(expected_vshale, 0.0, 1.0)

    result = shale_volume.vshale_larionov_old(gr=gr, grmin=grmin, grmax=grmax)
    assert np.isclose(result, expected_vshale, rtol=1e-6)

def test_vshale_larionov_shaliness_scalar():
    gr = 120.0
    grmin = 0.0
    grmax = 150.0

    expected_igr = (gr - grmin) / (grmax - grmin)
    expected_vshale = 0.083 * (2. ** (3.71 * expected_igr) - 1)
    expected_vshale = np.clip(expected_vshale, 0.0, 1.0)

    result = shale_volume.vshale_larionov(gr=gr, grmin=grmin, grmax=grmax)
    assert np.isclose(result, expected_vshale, rtol=1e-6)

def test_vshale_clavier_shaliness_scalar():
    gr = 120.0
    grmin = 0.0
    grmax = 150.0

    expected_igr = (gr - grmin) / (grmax - grmin)
    expected_vshale = 1.7 - 3.38 * (1 - expected_igr) + 2.12 * (1 - expected_igr) ** 2
    expected_vshale = np.clip(expected_vshale, 0.0, 1.0)

    result = shale_volume.vshale_clavier(gr=gr, grmin=grmin, grmax=grmax)
    assert np.isclose(result, expected_vshale, rtol=1e-6)

def test_vshale_stieber_shaliness_scalar():
    gr = 120.0
    grmin = 0.0
    grmax = 150.0

    expected_igr = (gr - grmin) / (grmax - grmin)
    expected_vshale = expected_igr / (3 - 2 * expected_igr)
    expected_vshale = np.clip(expected_vshale, 0.0, 1.0)

    result = shale_volume.vshale_stieber(gr=gr, grmin=grmin, grmax=grmax)
    assert np.isclose(result, expected_vshale, rtol=1e-6)

def test_vshale_neutron_density_shaliness_scalar():
    nphi = 0.25
    rhob = 2.4
    clean_n = -0.15,
    clean_d = 2.65,
    fluid_n = 1.00,
    fluid_d = 1.10,
    clay_n = 0.47,
    clay_d = 2.71

    x1 = (fluid_d - clean_d) * (nphi - clean_n)
    x2 = (clay_d - clean_d) * (clay_n - clean_n)
    x3 = (clay_d - clean_d) * (rhob - clean_d)
    x4 = (fluid_d - clean_d) * (fluid_n - clean_n)
    expected_vshale = (x1 - x3) / (x2 - x4)
    expected_vshale = np.clip(expected_vshale, 0.0, 1.0)

    result = shale_volume.vshale_neu_den(nphi=nphi, rhob=rhob, clean_n=clean_n, clean_d=clean_d, fluid_n=fluid_n, fluid_d=fluid_d, clay_n=clay_n, clay_d=clay_d)    
    assert np.isclose(result, expected_vshale, rtol=1e-6)

def test_vshale_nrm_shaliness_scalar():
    phit = 0.25
    phie = 0.10

    cbw = phie - phit
    vshale = cbw / phit
    expected_vshale = np.clip(vshale, 0.0, 1.0)
    result = shale_volume.vshale_nrm(phit=phit, phie=phie)
    assert np.isclose(result, expected_vshale, rtol=1e-6)