# %%
import pytest
import numpy as np
from hypothesis import given, strategies as st, assume

if __package__:
    from ..petrophysics.porosity import permeability
else:
    from stoneforge.petrophysics import permeability

# -------------------------------------------------------------------------------------------------------------- #
# test functions

@given(
    phi=st.floats(min_value=0.05, max_value=0.4),
    sw=st.floats(min_value=0.05, max_value=1.0),
    resd=st.floats(min_value=1.0, max_value=5000.0),
    ress=st.floats(min_value=0.1, max_value=1000.0),
)
def test_permeability_non_negative(phi, sw, resd, ress):
    rw = 0.02
    rhow = 1.10
    rhoo = 10000.0
    inzone = 0.75
    hd = 0.5

    models = [
        lambda: permeability.timur(phi, sw),
        lambda: permeability.coates(phi, sw),
        lambda: permeability.coates_dumanoir(resd, phi, hd, rw),
        lambda: permeability.tixier(resd, ress, rw, rhow, rhoo, inzone),
    ]

    for model in models:
        k = model()
        assert np.isnan(k) or k >= 0.0


@given(
    phi1=st.floats(min_value=0.05, max_value=0.15),
    phi2=st.floats(min_value=0.25, max_value=0.4),
)
def test_timur_porosity_monotonic(phi1, phi2):
    sw = 0.5
    assume(phi2 > phi1)

    k1 = permeability.timur(phi1, sw)
    k2 = permeability.timur(phi2, sw)

    assert k2 >= k1


@given(
    sw1=st.floats(min_value=0.05, max_value=0.3),
    sw2=st.floats(min_value=0.5, max_value=1.0),
)
def test_timur_sw_monotonic(sw1, sw2):
    phi = 0.25
    assume(sw2 > sw1)

    k1 = permeability.timur(phi, sw1)
    k2 = permeability.timur(phi, sw2)

    assert k2 <= k1


@given(
    phi1=st.floats(min_value=0.05, max_value=0.15),
    phi2=st.floats(min_value=0.25, max_value=0.4),
    sw1=st.floats(min_value=0.05, max_value=0.3),
    sw2=st.floats(min_value=0.5, max_value=1.0),
)
def test_coates_monotonicity(phi1, phi2, sw1, sw2):
    assume(phi2 > phi1)
    assume(sw2 > sw1)

    k_low_phi = permeability.coates(phi1, sw1)
    k_high_phi = permeability.coates(phi2, sw1)
    k_high_sw = permeability.coates(phi2, sw2)

    assert k_high_phi >= k_low_phi
    assert k_high_sw <= k_high_phi


@given(
    res=st.floats(min_value=10.0, max_value=5000.0),
)
def test_tixier_zero_contrast_limit(res):
    rw = 0.02
    rhow = 1.10
    rhoo = 10000.0
    inzone = 0.75

    k = permeability.tixier(res, res, rw, rhow, rhoo, inzone)

    assert np.isclose(k, 0.0)


@given(
    delta1=st.floats(min_value=1.0, max_value=20.0),
    delta2=st.floats(min_value=50.0, max_value=500.0),
)
def test_tixier_monotonic_contrast(delta1, delta2):
    rw = 0.02
    rhow = 1.10
    rhoo = 10000.0
    inzone = 0.75

    resd1 = 50.0 + delta1
    resd2 = 50.0 + delta2
    ress = 50.0

    k1 = permeability.tixier(resd1, ress, rw, rhow, rhoo, inzone)
    k2 = permeability.tixier(resd2, ress, rw, rhow, rhoo, inzone)

    assert k2 >= k1


@given(
    phi1=st.floats(min_value=0.05, max_value=0.15),
    phi2=st.floats(min_value=0.25, max_value=0.4),
)
def test_coates_dumanoir_porosity_effect(phi1, phi2):
    resd = 100.0
    rw = 0.02
    hd = 0.5

    assume(phi2 > phi1)

    k1 = permeability.coates_dumanoir(resd, phi1, hd, rw)
    k2 = permeability.coates_dumanoir(resd, phi2, hd, rw)

    assert k2 >= k1


@given(
    phi=st.floats(min_value=0.15, max_value=0.35),
    sw1=st.floats(min_value=0.05, max_value=0.3),
    sw2=st.floats(min_value=0.6, max_value=1.0),
)
def test_cross_model_sw_trend(phi, sw1, sw2):
    assume(sw2 > sw1)

    k_timur_1 = permeability.timur(phi, sw1)
    k_timur_2 = permeability.timur(phi, sw2)

    k_coates_1 = permeability.coates(phi, sw1)
    k_coates_2 = permeability.coates(phi, sw2)

    assert k_timur_2 <= k_timur_1
    assert k_coates_2 <= k_coates_1
