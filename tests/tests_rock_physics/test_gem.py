
import numpy as np
import pytest

from stoneforge.rock_physics.gem import (
    constant_cement,
    contact_cement,
    gem,
    gem_model,
    hertz_mindlin,
    soft_sand,
    stiff_sand,
)

# Just bounds testing
def test_soft_sand():
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="soft_sand",
               p=27*10**6)[0] < 10**15
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="soft_sand",
               p=27*10**6)[1] < 10**15
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="soft_sand",
               p=27*10**6)[0] > 0.0
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="soft_sand",
               p=27*10**6)[1] > 0.0


def test_stiff_sand():
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="stiff_sand",
               p=27*10**6)[0] < 10**15
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="stiff_sand",
               p=27*10**6)[1] < 10**15
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="stiff_sand",
               p=27*10**6)[0] > 0.0
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="stiff_sand",
               p=27*10**6)[1] > 0.0


def test_contact_cement():
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="contact_cement",
               kc=36*10**9, gc=45*10**9)[0] < 10**15
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="contact_cement",
               kc=36*10**9, gc=45*10**9)[1] < 10**15
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="contact_cement",
               kc=36*10**9, gc=45*10**9)[0] > 0.0
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="contact_cement",
               kc=36*10**9, gc=45*10**9)[1] > 0.0

def test_constant_cement():
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="constant_cement",
               kc=36*10**9, gc=45*10**9, phib=0.34)[0] < 10**15
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="constant_cement",
               kc=36*10**9, gc=45*10**9, phib=0.34)[1] < 10**15
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="constant_cement",
               kc=36*10**9, gc=45*10**9, phib=0.34)[0] > 0.0
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="constant_cement",
               kc=36*10**9, gc=45*10**9, phib=0.34)[1] > 0.0


def test_hertz_mindlin_returns_known_reference_values():
    bulk, shear = hertz_mindlin(k=36.0, g=45.0, n=5.0, phic=0.4, p=0.027)

    assert np.isclose(bulk, 1.4623084043642425)
    assert np.isclose(shear, 2.153581468245521)


def test_soft_and_stiff_sand_accept_arrays_and_soft_is_lower():
    phi = np.array([0.0, 0.2, 0.4])

    soft_bulk, soft_shear = soft_sand(
        k=36.0,
        g=45.0,
        phi=phi,
        phic=0.4,
        n=5.0,
        p=0.027,
    )
    stiff_bulk, stiff_shear = stiff_sand(
        k=36.0,
        g=45.0,
        phi=phi,
        phic=0.4,
        n=5.0,
        p=0.027,
    )

    assert soft_bulk.shape == phi.shape
    assert soft_shear.shape == phi.shape
    np.testing.assert_allclose(soft_bulk[-1], stiff_bulk[-1])
    np.testing.assert_allclose(soft_shear[-1], stiff_shear[-1])
    assert np.all(soft_bulk[1:-1] < stiff_bulk[1:-1])
    assert np.all(soft_shear[1:-1] < stiff_shear[1:-1])


def test_contact_cement_supports_grain_contact_deposition():
    surface_bulk, surface_shear = contact_cement(
        k=36.0,
        g=45.0,
        phi=np.array([0.2, 0.35]),
        phic=0.4,
        n=5.0,
        kc=36.0,
        gc=45.0,
        deposition_type="grain_surface",
    )
    contact_bulk, contact_shear = contact_cement(
        k=36.0,
        g=45.0,
        phi=np.array([0.2, 0.35]),
        phic=0.4,
        n=5.0,
        kc=36.0,
        gc=45.0,
        deposition_type="grain_contact",
    )

    assert np.all(surface_bulk > 0)
    assert np.all(surface_shear > 0)
    assert np.all(contact_bulk > 0)
    assert np.all(contact_shear > 0)
    assert not np.allclose(surface_bulk, contact_bulk)
    assert not np.allclose(surface_shear, contact_shear)


def test_constant_cement_array_uses_cement_and_soft_domains():
    phi = np.array([0.1, 0.2, 0.34, 0.38])

    bulk, shear = constant_cement(
        k=36.0,
        g=45.0,
        phi=phi,
        phic=0.4,
        n=5.0,
        kc=36.0,
        gc=45.0,
        phib=0.34,
    )

    cement_bulk, cement_shear = contact_cement(
        k=36.0,
        g=45.0,
        phi=phi,
        phic=0.4,
        n=5.0,
        kc=36.0,
        gc=45.0,
    )

    np.testing.assert_allclose(bulk[phi >= 0.34], cement_bulk[phi >= 0.34])
    np.testing.assert_allclose(shear[phi >= 0.34], cement_shear[phi >= 0.34])
    assert np.all(bulk[phi < 0.34] > cement_bulk[phi < 0.34])
    assert np.all(shear[phi < 0.34] > cement_shear[phi < 0.34])


def test_constant_cement_scalar_branches():
    cement_branch = constant_cement(
        k=36.0,
        g=45.0,
        phi=0.36,
        phic=0.4,
        n=5.0,
        kc=36.0,
        gc=45.0,
        phib=0.34,
    )
    soft_branch = constant_cement(
        k=36.0,
        g=45.0,
        phi=0.2,
        phic=0.4,
        n=5.0,
        kc=36.0,
        gc=45.0,
        phib=0.34,
    )

    assert np.asarray(cement_branch[0]).shape == ()
    assert np.asarray(cement_branch[1]).shape == ()
    assert np.asarray(soft_branch[0]).shape == ()
    assert np.asarray(soft_branch[1]).shape == ()
    assert soft_branch[0] > cement_branch[0]
    assert soft_branch[1] > cement_branch[1]


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("soft_sand", {"p": 0.027}),
        ("stiff_sand", {"p": 0.027}),
        ("contact_cement", {"kc": 36.0, "gc": 45.0}),
        ("constant_cement", {"kc": 36.0, "gc": 45.0, "phib": 0.34}),
    ],
)
def test_gem_model_returns_porosity_curves_for_all_methods(method, kwargs):
    bulk, shear = gem_model(
        k=36.0,
        g=45.0,
        phic=0.4,
        n=5.0,
        method=method,
        **kwargs,
    )

    assert bulk.shape == (100,)
    assert shear.shape == (100,)
    assert np.all(np.isfinite(bulk))
    assert np.all(np.isfinite(shear))


def test_gem_and_gem_model_validate_method_and_required_arguments():
    with pytest.raises(ValueError, match="Unsupported method"):
        gem(36.0, 45.0, 0.2, 0.4, 5.0, method="unknown")

    with pytest.raises(TypeError, match="Missing required arguments"):
        gem(36.0, 45.0, 0.2, 0.4, 5.0, method="soft_sand")

    with pytest.raises(ValueError, match="Unsupported method"):
        gem_model(36.0, 45.0, 0.4, 5.0, method="unknown")

    with pytest.raises(TypeError, match="Missing required arguments"):
        gem_model(36.0, 45.0, 0.4, 5.0, method="constant_cement", kc=36.0)
