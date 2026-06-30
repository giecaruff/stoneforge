import numpy as np
from stoneforge.rock_physics import inclusion

def test_theta_and_f_match_reference_calculation():
    alpha = 0.5

    theta = inclusion.get_theta(alpha)
    f_value = inclusion.get_f(alpha, theta)

    assert np.isclose(theta, 0.47279971743743016)
    assert np.isclose(f_value, -0.19386694922923652)


def test_abr_uses_poisson_ratio_for_solid_phase():
    a_value, b_value, r_value = inclusion.ABR(k=2.0, g=1.0, ks=36.0, gs=45.0)

    assert np.isclose(a_value, -0.9777777777777777)
    assert np.isclose(b_value, 0.011111111111111112)
    assert np.isclose(r_value, 0.46875)


def test_pq_returns_finite_shape_factors():
    theta = inclusion.get_theta(0.5)
    f_value = inclusion.get_f(0.5, theta)

    p_value, q_value = inclusion.PQ(
        A=-0.5,
        B=0.1,
        R=0.25,
        theta=theta,
        f=f_value,
    )

    assert np.isfinite(p_value)
    assert np.isfinite(q_value)
    assert np.isclose(p_value, 1.1572577139086841)
    assert np.isclose(q_value, 1.3103218358579105)


def test_kuster_toksoz_returns_bulk_and_shear_arrays():
    phi = np.array([0.0, 0.1, 0.2])
    kuster_toksoz = getattr(inclusion, "Kuster_Toks\u00f6z")

    bulk, shear = kuster_toksoz(
        phi=phi,
        ks=np.array([36.0, 36.0, 36.0]),
        gs=np.array([45.0, 45.0, 45.0]),
        k=2.0,
        g=1.0,
        alpha=0.5,
    )

    assert bulk.shape == phi.shape
    assert shear.shape == phi.shape
    assert bulk[0] == 36.0
    assert shear[0] == 45.0
    assert np.all(np.diff(bulk) < 0)
    assert np.all(np.diff(shear) < 0)
