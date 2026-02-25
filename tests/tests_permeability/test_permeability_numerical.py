# %%
import pytest
import numpy as np

if __package__:
    from ..petrophysics.porosity import permeability
else:
    from stoneforge.petrophysics import permeability

# -------------------------------------------------------------------------------------------------------------- #
# test functions

def test_tixier_vectorized_with_nan():
    resd = np.array([80.0, 120.0, np.nan])
    ress = 50.0
    rw = 0.02
    rhow = 1.10
    rhoo = 10000.0
    inzone = 0.75

    result = permeability.tixier(resd, ress, rw, rhow, rhoo, inzone)

    assert isinstance(result, np.ndarray)
    assert np.isnan(result[2])
    assert np.all(result[:2] >= 0.0)
    

def test_tixier_monotonic_resistivity_contrast():
    resd = np.array([70.0, 100.0, 150.0])
    ress = 50.0
    rw = 0.02
    rhow = 1.10
    rhoo = 10000.0
    inzone = 0.75

    result = permeability.tixier(resd, ress, rw, rhow, rhoo, inzone)

    assert np.all(np.diff(result) >= 0.0)
    
    
def test_timur_vectorized_behavior():
    phi = np.array([0.15, 0.25, 0.35])
    sw = np.array([0.7, 0.5, 0.3])

    result = permeability.timur(phi, sw)

    assert isinstance(result, np.ndarray)
    assert np.all(result >= 0.0)
    
    
def test_timur_monotonic_porosity():
    phi = np.array([0.10, 0.20, 0.30])
    sw = 0.5

    result = permeability.timur(phi, sw)

    assert np.all(np.diff(result) >= 0.0)
    
    
def test_timur_monotonic_sw():
    phi = 0.25
    sw = np.array([0.3, 0.5, 0.8])

    result = permeability.timur(phi, sw)

    assert np.all(np.diff(result) <= 0.0)


def test_coates_dumanoir_vectorized():
    resd = np.array([50.0, 100.0, 200.0])
    phi = np.array([0.15, 0.25, 0.30])
    hd = 0.5
    rw = 0.02

    result = permeability.coates_dumanoir(resd, phi, hd, rw)

    assert isinstance(result, np.ndarray)
    assert np.all(result >= 0.0)
    
    
def test_coates_dumanoir_nan_behavior():
    resd = np.array([100.0, np.nan, 200.0])
    phi = np.array([0.25, 0.30, 0.20])
    hd = 0.5
    rw = 0.02

    result = permeability.coates_dumanoir(resd, phi, hd, rw)

    assert np.isnan(result[1])
    assert np.all(result[[0, 2]] >= 0.0)


def test_coates_vectorized_behavior():
    phi = np.array([0.15, 0.25, 0.35])
    sw = np.array([0.7, 0.5, 0.3])

    result = permeability.coates(phi, sw)

    assert isinstance(result, np.ndarray)
    assert np.all(result >= 0.0)


def test_coates_monotonic_sw():
    phi = 0.25
    sw = np.array([0.3, 0.5, 0.8])

    result = permeability.coates(phi, sw)

    assert np.all(np.diff(result) <= 0.0)


def test_permeability_models_porosity_effect():
    phi = np.array([0.15, 0.25, 0.35])
    sw = 0.5

    k_timur = permeability.timur(phi, sw)
    k_coates = permeability.coates(phi, sw)

    assert np.all(np.diff(k_timur) >= 0.0)
    assert np.all(np.diff(k_coates) >= 0.0)
