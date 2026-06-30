import pytest
import numpy as np
from stoneforge.pseudo_wells import pseudo_tools

def test_merge_lithology():
    litho_ref = [1, 1, 1, 2, 2, 3, 3, 3, 3]
    values, counts = pseudo_tools.merge_lithology(litho_ref)
    assert values == [1, 2, 3]
    assert counts == [3, 2, 4]

def test_log_statstics():
    log = np.array([10.0, 11.0, 12.0, 20.0, 21.0, 30.0])
    lito = np.array([1.0, 1.0, 1.0, 2.0, 2.0, np.nan])
    
    stats = pseudo_tools.log_statstics(log, lito)
    assert 1.0 in stats
    assert 2.0 in stats
    
    # 1.0: [10, 11, 12] -> mean=11.0
    assert np.isclose(stats[1.0][0], 11.0)
    assert np.isclose(stats[1.0][1], np.std([10.0, 11.0, 12.0]))
    
    # 2.0: [20, 21] -> mean=20.5
    assert np.isclose(stats[2.0][0], 20.5)
    assert np.isclose(stats[2.0][1], np.std([20.0, 21.0]))

def test_synthetic_log():
    stats = {
        1.0: [10.0, 1.0],
        2.0: [20.0, 2.0]
    }
    lithology = [1.0, np.nan, 2.0]
    
    # Test reproducibility with seed
    res1 = pseudo_tools.synthetic_log(stats, lithology, seed=42)
    res2 = pseudo_tools.synthetic_log(stats, lithology, seed=42)
    assert len(res1) == 3
    assert np.isnan(res1[1])
    assert np.isnan(res2[1])
    assert np.isclose(res1[0], res2[0])
    assert np.isclose(res1[2], res2[2])
    
    # Test with different seed
    res3 = pseudo_tools.synthetic_log(stats, lithology, seed=100)
    assert not np.isclose(res1[0], res3[0])

def test_moving_average():
    curve = [1.0, 2.0, 3.0, 4.0, 5.0]
    # Test with even step (gets adjusted to step + step%2 = 2)
    smooth_even = pseudo_tools.moving_average(curve, step=2)
    expected_even = [4/3, 2.0, 3.0, 4.0, 14/3]
    assert np.allclose(smooth_even, expected_even)
    
    # Test with odd step (gets adjusted to step + step%2 = 3)
    # step=3 -> step%2 = 1 -> adjusted step = 4
    # extended_curve: pad size = 2 on each edge
    # padding edge: [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0]
    # window_shape = 5
    # windows:
    # [1, 1, 1, 2, 3] -> mean=1.6
    # [1, 1, 2, 3, 4] -> mean=2.2
    # [1, 2, 3, 4, 5] -> mean=3.0
    # [2, 3, 4, 5, 5] -> mean=3.8
    # [3, 4, 5, 5, 5] -> mean=4.4
    smooth_odd = pseudo_tools.moving_average(curve, step=3)
    expected_odd = [1.6, 2.2, 3.0, 3.8, 4.4]
    assert np.allclose(smooth_odd, expected_odd)

def test_gamma_calc():
    dif_curve = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    depth = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    
    gamma, depth_intervals = pseudo_tools.gamma_calc(dif_curve, depth, step=3)
    assert len(gamma) == 3
    assert len(depth_intervals) == 3
    
    # For st=0:
    # diffs = 0, gamma_value = [0, 0, 0, 0, 0] -> sum = 0
    assert np.isclose(gamma[0], 0.0)
    assert np.isclose(depth_intervals[0], 0.0)
    
    # For st=1:
    # i from 0 to 3: (dif_curve[i+1] - dif_curve[i])**2 = 1.0
    # gamma_value = [1.0, 1.0, 1.0, 1.0]
    # gamma = sum / (2 * 4) = 4.0 / 8.0 = 0.5
    assert np.isclose(gamma[1], 0.5)
    assert np.isclose(depth_intervals[1], 1.0)

def test_adjustment():
    dept = np.array([0.0, 1.0, 2.0])
    
    # Exponential mode
    cov_matrix_exp = pseudo_tools.adjustment(dept, a=2.0, C1=1.5, C0=0.5, mode="exponential")
    assert cov_matrix_exp.shape == (3, 3)
    
    # Check diagonals (x = 0) -> C0 + C1 = 2.0
    assert np.isclose(cov_matrix_exp[0, 0], 2.0)
    assert np.isclose(cov_matrix_exp[1, 1], 2.0)
    assert np.isclose(cov_matrix_exp[2, 2], 2.0)
    
    # Check off-diagonals (x = 1.0) -> C1 * exp(-3 * 1 / 2) = 1.5 * exp(-1.5)
    expected_off_1 = 1.5 * np.exp(-1.5)
    assert np.isclose(cov_matrix_exp[0, 1], expected_off_1)
    assert np.isclose(cov_matrix_exp[1, 0], expected_off_1)
    
    # Invalid mode raises UnboundLocalError
    with pytest.raises(UnboundLocalError):
        pseudo_tools.adjustment(dept, a=2.0, C1=1.5, C0=0.5, mode="invalid_mode")

def test_cov_matrix():
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cov = pseudo_tools.cov_matrix(M)
    assert cov.shape == (2, 2)
    # Check np.cov compatibility
    expected = np.cov(M[0], M[1])
    assert np.isclose(cov[0, 1], expected[0, 1])
    assert np.isclose(cov[1, 0], expected[1, 0])
    assert np.isclose(cov[0, 0], expected[0, 0])
    assert np.isclose(cov[1, 1], expected[1, 1])
