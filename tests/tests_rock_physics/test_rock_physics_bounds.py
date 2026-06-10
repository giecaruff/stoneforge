import numpy as np

from stoneforge.rock_physics.rock_physics_bounds import hill, reuss, voigt


def test_reuss_voigt_and_hill_for_scalar_mineral_mix():
    fractions = np.array([0.25, 0.75])
    moduli = np.array([10.0, 40.0])

    assert np.isclose(reuss(fractions, moduli), 22.857142857142858)
    assert np.isclose(voigt(fractions, moduli), 32.5)
    assert np.isclose(hill(fractions, moduli), 27.67857142857143)


def test_bounds_operate_columnwise_for_logs():
    fractions = np.array([[0.2, 0.4], [0.8, 0.6]])
    moduli = np.array([[15.0, 20.0], [30.0, 50.0]])

    expected_reuss = 1 / np.sum(fractions / moduli, axis=0)
    expected_voigt = np.sum(fractions * moduli, axis=0)

    np.testing.assert_allclose(reuss(fractions, moduli), expected_reuss)
    np.testing.assert_allclose(voigt(fractions, moduli), expected_voigt)
    np.testing.assert_allclose(
        hill(fractions, moduli),
        (expected_reuss + expected_voigt) / 2,
    )
