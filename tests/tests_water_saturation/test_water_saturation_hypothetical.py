# %%
import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
import numpy as np

if __package__:
    from ..petrophysics.porosity import water_saturation
else:
    from stoneforge.petrophysics import water_saturation

# -------------------------------------------------------------------------------------------------------------- #
# test functions

@given(
    rt=st.floats(min_value=1.0, max_value=2000.0),
    phi=st.floats(min_value=0.05, max_value=0.4),
    rw=st.floats(min_value=0.005, max_value=0.2),
    vsh=st.floats(min_value=0.0, max_value=0.6),
    rsh=st.floats(min_value=1.0, max_value=50.0),
)
def test_water_saturation_bounds(rt, phi, rw, vsh, rsh):
    models = [
        lambda: water_saturation.archie(rt=rt, phi=phi, rw=rw),
        lambda: water_saturation.simandoux(rt=rt, phi=phi, rw=rw, vsh=vsh, rsh=rsh),
        lambda: water_saturation.indonesia(rt=rt, phi=phi, rw=rw, vsh=vsh, rsh=rsh),
        lambda: water_saturation.fertl(rt=rt, phi=phi, rw=rw, vsh=vsh),
    ]

    for model in models:
        sw = model()
        assert np.isnan(sw) or (0.0 <= sw <= 1.0)


@given(
    rt1=st.floats(min_value=2.0, max_value=50.0),
    rt2=st.floats(min_value=100.0, max_value=2000.0),
)
def test_archie_monotonic_rt(rt1, rt2):
    phi = 0.25
    rw = 0.02

    assume(rt2 > rt1)

    sw1 = water_saturation.archie(rt=rt1, phi=phi, rw=rw)
    sw2 = water_saturation.archie(rt=rt2, phi=phi, rw=rw)

    assert sw2 <= sw1


@given(
    phi1=st.floats(min_value=0.05, max_value=0.15),
    phi2=st.floats(min_value=0.25, max_value=0.4),
)
def test_archie_porosity_effect(phi1, phi2):
    rt = 200.0
    rw = 0.02

    assume(phi2 > phi1)

    sw1 = water_saturation.archie(rt=rt, phi=phi1, rw=rw)
    sw2 = water_saturation.archie(rt=rt, phi=phi2, rw=rw)

    assert sw2 <= sw1


@given(
    rt=st.floats(min_value=500.0, max_value=5000.0),
)
def test_archie_high_rt_limit(rt):
    phi = 0.25
    rw = 0.02

    sw = water_saturation.archie(rt=rt, phi=phi, rw=rw)

    assert sw < 0.1


@given(
    rt=st.floats(min_value=10.0, max_value=1000.0),
    phi=st.floats(min_value=0.1, max_value=0.4),
    rw=st.floats(min_value=0.005, max_value=0.2),
)
def test_simandoux_reduces_to_archie(rt, phi, rw):
    sw_archie = water_saturation.archie(rt=rt, phi=phi, rw=rw)
    sw_sim = water_saturation.simandoux(
        rt=rt, phi=phi, rw=rw, vsh=0.0, rsh=10.0
    )

    assert np.isclose(sw_sim, sw_archie, rtol=1e-5)


# TODO: verify commented tests conditions to enable then.
#@given(
#    rt=st.floats(min_value=10.0, max_value=1000.0),
#    phi=st.floats(min_value=0.05, max_value=0.4),
#    rw=st.floats(min_value=0.005, max_value=0.2),
#)
#def test_indonesia_high_shale_limit(rt, phi, rw):
#    sw = water_saturation.indonesia(
#        rt=rt, phi=phi, rw=rw, vsh=0.99, rsh=4.0
#    )

#    assert sw > 0.5 # Should be greather then 0.7 maybe


#@given(
#    vsh1=st.floats(min_value=0.0, max_value=0.2),
#    vsh2=st.floats(min_value=0.3, max_value=0.6),
#)
#def test_fertl_shale_increases_sw(vsh1, vsh2):
#    rt = 200.0
#    phi = 0.25
#    rw = 0.02

#    assume(vsh2 > vsh1)

#    sw1 = water_saturation.fertl(rt=rt, phi=phi, rw=rw, vsh=vsh1)
#    sw2 = water_saturation.fertl(rt=rt, phi=phi, rw=rw, vsh=vsh2)

#    assert sw2 >= sw1


#@given(
#    rt=st.floats(min_value=50.0, max_value=500.0),
#    phi=st.floats(min_value=0.15, max_value=0.35),
#    rw=st.floats(min_value=0.01, max_value=0.1),
#)
#def test_shaly_models_increase_sw(rt, phi, rw):
#    sw_archie = water_saturation.archie(rt=rt, phi=phi, rw=rw)

#    sw_sim = water_saturation.simandoux(
#        rt=rt, phi=phi, rw=rw, vsh=0.2, rsh=4.0
#    )
#    sw_ind = water_saturation.indonesia(
#        rt=rt, phi=phi, rw=rw, vsh=0.2, rsh=4.0
#    )
#    sw_fertl = water_saturation.fertl(
#        rt=rt, phi=phi, rw=rw, vsh=0.2
#    )

#    assert sw_archie <= sw_sim
#    assert sw_archie <= sw_ind
#    assert sw_archie <= sw_fertl
