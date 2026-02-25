# %%
import pytest
import numpy as np

if __package__:
    from ..petrophysics.porosity import water_saturation
else:
    from stoneforge.petrophysics import water_saturation

# -------------------------------------------------------------------------------------------------------------- #
# test functions

def test_archie_vectorized_with_nan():
    rt = np.array([50.0, 200.0, np.nan])
    phi = np.array([0.20, 0.25, 0.30])
    rw = 0.02
    a = 1.0
    m = 2.0
    n = 2.0

    result = water_saturation.archie(rt=rt, phi=phi, rw=rw, a=a, m=m, n=n)

    assert isinstance(result, np.ndarray)
    assert np.isnan(result[2])
    assert np.all(result[:2] >= 0.0)
    assert np.all(result[:2] <= 1.0)


def test_archie_vectorized_monotonic_rt():
    rt = np.array([20.0, 50.0, 200.0])
    phi = 0.25
    rw = 0.02
    a = 1.0
    m = 2.0
    n = 2.0

    result = water_saturation.archie(rt=rt, phi=phi, rw=rw, a=a, m=m, n=n)

    assert np.all(np.diff(result) <= 0.0)


def test_simandoux_vectorized_bounds():
    rt = np.array([50.0, 200.0, 500.0])
    phi = np.array([0.20, 0.25, 0.30])
    rw = 0.02
    a = 1.0
    m = 2.0
    n = 2.0
    vsh = 0.15
    rsh = 4.0

    result = water_saturation.simandoux(
        rt=rt, phi=phi, rw=rw, a=a, m=m, n=n, vsh=vsh, rsh=rsh
    )

    assert isinstance(result, np.ndarray)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_indonesia_vectorized_broadcasting():
    rt = np.array([30.0, 100.0, 300.0])
    phi = 0.25
    rw = 0.02
    a = 1.0
    m = 2.0
    n = 2.0
    vsh = np.array([0.05, 0.15, 0.30])
    rsh = 4.0

    result = water_saturation.indonesia(
        rt=rt, phi=phi, rw=rw, a=a, m=m, n=n, vsh=vsh, rsh=rsh
    )

    assert isinstance(result, np.ndarray)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_indonesia_vectorized_nan():
    rt = np.array([100.0, np.nan, 300.0])
    phi = np.array([0.25, 0.30, 0.20])
    rw = 0.02
    a = 1.0
    m = 2.0
    n = 2.0
    vsh = 0.1
    rsh = 4.0

    result = water_saturation.indonesia(
        rt=rt, phi=phi, rw=rw, a=a, m=m, n=n, vsh=vsh, rsh=rsh
    )

    assert np.isnan(result[1])
    assert np.all(result[[0, 2]] >= 0.0)
    assert np.all(result[[0, 2]] <= 1.0)


def test_fertl_vectorized_behavior():
    rt = np.array([20.0, 100.0, 300.0])
    phi = np.array([0.18, 0.25, 0.30])
    rw = 0.02
    a = 1.0
    m = 2.0
    vsh = 0.2
    alpha = 0.5

    result = water_saturation.fertl(
        rt=rt, phi=phi, rw=rw, a=a, m=m, vsh=vsh, alpha=alpha
    )

    assert isinstance(result, np.ndarray)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_water_saturation_models_monotonic_rt():
    rt = np.array([30.0, 100.0, 300.0])
    phi = 0.25
    rw = 0.02

    archie = water_saturation.archie(rt=rt, phi=phi, rw=rw)
    sim = water_saturation.simandoux(rt=rt, phi=phi, rw=rw, vsh=0.1, rsh=4.0)
    indo = water_saturation.indonesia(rt=rt, phi=phi, rw=rw, vsh=0.1, rsh=4.0)

    assert np.all(np.diff(archie) <= 0.0)
    assert np.all(np.diff(sim) <= 0.0)
    assert np.all(np.diff(indo) <= 0.0)
