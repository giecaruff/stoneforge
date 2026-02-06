# %%
import pytest
import numpy as np

if __package__:
    from ..petrophysics.porosity import shale_volume
else:
    from stoneforge.petrophysics import shale_volume

# -------------------------------------------------------------------------------------------------------------- #
# test functions

def test_gammarayindex_vectorized_with_nan():
    gr = np.array([50.0, 120.0, np.nan])
    grmin = 0.0
    grmax = 150.0

    expected = (gr - grmin) / (grmax - grmin)
    result = shale_volume.gammarayindex(gr=gr, grmin=grmin, grmax=grmax)

    assert isinstance(result, np.ndarray)
    assert np.isnan(result[2])
    assert np.allclose(result[:2], expected[:2], rtol=1e-6)
    
def test_vshale_linear_vectorized_clipping():
    gr = np.array([-10.0, 75.0, 200.0])
    grmin = 0.0
    grmax = 150.0

    result = shale_volume.vshale_linear(gr=gr, grmin=grmin, grmax=grmax)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    
def test_vshale_larionov_old_vectorized_monotonic():
    gr = np.array([30.0, 80.0, 130.0])
    grmin = 0.0
    grmax = 150.0

    result = shale_volume.vshale_larionov_old(gr=gr, grmin=grmin, grmax=grmax)

    assert isinstance(result, np.ndarray)
    assert np.all(np.diff(result) >= 0.0)
    
def test_vshale_larionov_vectorized_bounds():
    gr = np.array([20.0, 100.0, 180.0])
    grmin = 0.0
    grmax = 150.0

    result = shale_volume.vshale_larionov(gr=gr, grmin=grmin, grmax=grmax)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    
def test_vshale_clavier_vectorized_nan():
    gr = np.array([60.0, np.nan, 120.0])
    grmin = 0.0
    grmax = 150.0

    result = shale_volume.vshale_clavier(gr=gr, grmin=grmin, grmax=grmax)

    assert np.isnan(result[1])
    assert np.all(result[[0, 2]] >= 0.0)
    assert np.all(result[[0, 2]] <= 1.0)
    
def test_vshale_stieber_vectorized_behavior():
    gr = np.array([10.0, 70.0, 140.0])
    grmin = 0.0
    grmax = 150.0

    result = shale_volume.vshale_stieber(gr=gr, grmin=grmin, grmax=grmax)

    assert isinstance(result, np.ndarray)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    
def test_vshale_neutron_density_vectorized():
    nphi = np.array([0.20, 0.30, np.nan])
    rhob = np.array([2.35, 2.45, 2.50])

    result = shale_volume.vshale_neu_den(
        nphi=nphi,
        rhob=rhob,
        clean_n=-0.15,
        clean_d=2.65,
        fluid_n=1.00,
        fluid_d=1.10,
        clay_n=0.47,
        clay_d=2.71,
    )

    assert np.isnan(result[2])
    assert np.all(result[:2] >= 0.0)
    assert np.all(result[:2] <= 1.0)
    
def test_vshale_nrm_vectorized_behavior():
    phit = np.array([0.30, 0.25, 0.20])
    phie = np.array([0.15, 0.10, 0.05])

    result = shale_volume.vshale_nrm(phit=phit, phie=phie)

    assert isinstance(result, np.ndarray)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)