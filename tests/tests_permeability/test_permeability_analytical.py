# %%
import pytest
import numpy as np

if __package__:
    from ..petrophysics.porosity import permeability
else:
    from stoneforge.petrophysics import permeability

# -------------------------------------------------------------------------------------------------------------- #
# test functions

def test_tixier_scalar():
    resd = 100.0
    ress = 50.0
    rw = 0.02
    rhow = 1.10
    rhoo = 10000.00
    inzone = 0.75

    expected_k = 20 * ((2.3 / (rw * (rhow - rhoo))) * ((resd - ress) / inzone)) ** 2

    calculated_k = permeability.tixier(resd, ress, rw, rhow, rhoo, inzone)

    assert np.isclose(calculated_k, expected_k), f"Expected {expected_k}, got {calculated_k}"
    

def test_timur_scalar():
    phi = 0.25
    sw = 0.5

    expected_k = (93 * (phi ** 2.2) / (sw)) ** 2

    calculated_k = permeability.timur(phi, sw)

    assert np.isclose(calculated_k, expected_k), f"Expected {expected_k}, got {calculated_k}"
    
def test_coates_dumanoir_scalar():
    resd = 100.0
    phi = 0.25
    hd = 0.5
    rw = 0.02

    calculated_k = permeability.coates_dumanoir(resd, phi, hd, rw)
    
    # Calculating C constant
    c = 23 + 465 * hd - 188 * hd*hd
    
    # Calculating W constant
    w = np.sqrt((((np.log10(rw / resd) + 2.2) ** 2) / 2.0) + (3.75 - phi))

    # Final permeability
    expected_k = ((c * phi ** (2 * w))/((w ** 4) * (rw / resd))) ** 2

    assert np.isclose(calculated_k, expected_k), f"Expected {expected_k}, got {calculated_k}"

def test_coates_scalar():
    resd = 100.0
    phi = 0.25
    hd = 0.5
    rw = 0.02

    calculated_k = permeability.coates(resd, phi, hd, rw)
    
    # Calculating C constant
    c = 23 + 465 * hd - 188 * hd*hd
    
    # Calculating W constant
    w = np.sqrt((((np.log10(rw / resd) + 2.2) ** 2) / 2.0) + (3.75 - phi))

    # Final permeability
    expected_k = ((c * phi ** (2 * w))/((w ** 4) * (rw / resd))) ** 2

    assert np.isclose(calculated_k, expected_k), f"Expected {expected_k}, got {calculated_k}"