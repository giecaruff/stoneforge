import numpy as np
import pytest

from stoneforge.rock_physics.elastic_constants import (
    bulk_modulus,
    compressional_modulus,
    compressional_wave_velocity,
    poisson,
    shear_modulus,
    shear_wave_velocity,
)


def test_basic_elastic_moduli_equations_for_arrays():
    rho = np.array([2.2, 2.5])
    vp = np.array([3.0, 4.0])
    vs = np.array([1.5, 2.0])

    np.testing.assert_allclose(
        bulk_modulus(rho, vp, vs),
        rho * (vp**2 - (4 / 3) * vs**2),
    )
    np.testing.assert_allclose(compressional_modulus(rho, vp), rho * vp**2)
    np.testing.assert_allclose(shear_modulus(rho, vs), rho * vs**2)


def test_wave_velocity_equations():
    rho = np.array([2.0, 4.0])
    shear = np.array([18.0, 64.0])

    np.testing.assert_allclose(shear_wave_velocity(rho, shear), [3.0, 4.0])
    np.testing.assert_allclose(
        compressional_wave_velocity(
            method="rhob_and_g_and_k",
            rhob=rho,
            g=np.array([6.0, 12.0]),
            k=np.array([15.0, 48.0]),
        ),
        [np.sqrt(11.5), 4.0],
    )
    np.testing.assert_allclose(
        compressional_wave_velocity(
            method="rhob_and_m",
            rhob=rho,
            m=np.array([18.0, 64.0]),
        ),
        [3.0, 4.0],
    )


def test_compressional_wave_velocity_validates_method_and_arguments():
    with pytest.raises(ValueError, match="Unsupported method"):
        compressional_wave_velocity(method="invalid")

    with pytest.raises(TypeError, match="Missing required arguments"):
        compressional_wave_velocity(method="rhob_and_m", rhob=2.5)


def test_poisson_methods_return_expected_values():
    assert np.isclose(
        poisson(method="k_and_g", k=36.0, g=45.0),
        0.0588235294,
    )
    assert np.isclose(poisson(method="vp_and_vs", vp=3.0, vs=1.5), 1 / 3)
    assert np.isclose(poisson(method="e_and_k", e=30.0, k=20.0), 0.25)


def test_poisson_validates_method_and_arguments():
    with pytest.raises(ValueError, match="Unsupported method"):
        poisson(method="invalid")

    with pytest.raises(TypeError, match="Missing required arguments"):
        poisson(method="k_and_g", k=36.0)
