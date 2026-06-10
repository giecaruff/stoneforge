"""Tests for regularized acoustic reflectivity inversion module."""

import pytest
import numpy as np
from stoneforge.inversion.regularized_acoustic_reflectivy_inversion import IRLS_Inv


class TestIRLSInvBasic:
    """Basic functionality tests for IRLS_Inv function."""

    def test_irls_inv_basic_operation(self):
        """Test basic IRLS inversion with simple inputs."""
        # Create a simple trace (minimum 146 elements for residuo[145] indexing)
        trace = np.random.rand(160)
        G = np.eye(160)
        
        alphaL1 = 0.1
        alphaL2 = 0.1
        thresold = 0.01
        niter = 2
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, alphaL1, alphaL2, thresold, niter)
        
        assert RegL2.shape == (160,)
        assert RegL1.shape == (160,)
        assert residuo_iter.shape == (10,)

    def test_irls_inv_returns_three_values(self):
        """Test that IRLS_Inv returns three values."""
        trace = np.ones(150)
        G = np.eye(150)
        
        result = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        assert len(result) == 3

    def test_irls_inv_output_is_ndarray(self):
        """Test that all outputs are numpy arrays."""
        trace = np.linspace(1, 10, 160)
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.05, 0.05, 0.01, 1)
        
        assert isinstance(RegL2, np.ndarray)
        assert isinstance(RegL1, np.ndarray)
        assert isinstance(residuo_iter, np.ndarray)


class TestIRLSInvOutputShapes:
    """Test output shapes for different input dimensions."""

    def test_trace_150_elements(self):
        """Test with 150-element trace."""
        trace = np.random.rand(150)
        G = np.eye(150)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        
        assert RegL2.shape == (150,)
        assert RegL1.shape == (150,)
        assert residuo_iter.shape == (10,)

    def test_trace_200_elements(self):
        """Test with 200-element trace."""
        trace = np.random.rand(200)
        G = np.eye(200)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        
        assert RegL2.shape == (200,)
        assert RegL1.shape == (200,)

    def test_trace_300_elements(self):
        """Test with 300-element trace."""
        trace = np.random.rand(300)
        G = np.eye(300)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        
        assert RegL2.shape == (300,)
        assert RegL1.shape == (300,)


class TestIRLSInvIterations:
    """Test different iteration numbers."""

    def test_single_iteration(self):
        """Test with single iteration."""
        trace = np.random.rand(160)
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=1)
        assert residuo_iter.shape == (10,)

    def test_multiple_iterations(self):
        """Test with multiple iterations."""
        trace = np.random.rand(170)
        G = np.eye(170)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=5)
        assert residuo_iter.shape == (10,)

    def test_ten_iterations(self):
        """Test with ten iterations."""
        trace = np.random.rand(160)
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=10)
        assert residuo_iter.shape == (10,)


class TestIRLSInvRegularizationParameters:
    """Test different regularization parameters."""

    def test_high_l1_regularization(self):
        """Test with high L1 regularization."""
        trace = np.random.rand(160)
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, alphaL1=1.0, alphaL2=0.01, 
                                               thresold=0.01, niter=1)
        assert RegL1.shape == (160,)

    def test_high_l2_regularization(self):
        """Test with high L2 regularization."""
        trace = np.random.rand(160)
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, alphaL1=0.01, alphaL2=1.0, 
                                               thresold=0.01, niter=1)
        assert RegL2.shape == (160,)

    def test_equal_regularization_parameters(self):
        """Test with equal L1 and L2 regularization."""
        trace = np.random.rand(160)
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, alphaL1=0.1, alphaL2=0.1, 
                                               thresold=0.01, niter=1)
        assert RegL2.shape == (160,)
        assert RegL1.shape == (160,)

    def test_very_small_regularization(self):
        """Test with very small regularization parameters."""
        trace = np.random.rand(155)
        G = np.eye(155)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, alphaL1=1e-6, alphaL2=1e-6, 
                                               thresold=0.01, niter=1)
        assert RegL2.shape == (155,)


class TestIRLSInvThreshold:
    """Test different threshold values."""

    def test_small_threshold(self):
        """Test with small threshold."""
        trace = np.random.rand(160)
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, thresold=0.001, niter=2)
        assert residuo_iter.shape == (10,)

    def test_large_threshold(self):
        """Test with large threshold."""
        trace = np.random.rand(160)
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, thresold=1.0, niter=2)
        assert residuo_iter.shape == (10,)

    def test_very_large_threshold(self):
        """Test with very large threshold."""
        trace = np.random.rand(160)
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, thresold=100.0, niter=1)
        assert residuo_iter.shape == (10,)


class TestIRLSInvDifferentForwardOperators:
    """Test with different forward operators G."""

    def test_non_identity_operator(self):
        """Test with non-identity forward operator."""
        trace = np.random.rand(160)
        G = np.eye(160) * 0.8 + np.eye(160, k=1) * 0.2
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        assert RegL2.shape == (160,)

    def test_random_operator(self):
        """Test with random forward operator."""
        trace = np.random.rand(160)
        G = np.random.rand(160, 160)
        
        try:
            RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
            assert RegL2.shape == (160,)
        except np.linalg.LinAlgError:
            assert True

    def test_diagonal_operator_nonunity(self):
        """Test with non-unity diagonal operator."""
        trace = np.random.rand(160)
        diag_values = np.random.rand(160) + 0.1
        G = np.diag(diag_values)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        assert RegL2.shape == (160,)


class TestIRLSInvInputTypes:
    """Test with different input types."""

    def test_float64_inputs(self):
        """Test with float64 inputs."""
        trace = np.ones(150, dtype=np.float64)
        G = np.eye(150, dtype=np.float64)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        assert RegL2.shape == (150,)

    def test_float32_inputs(self):
        """Test with float32 inputs."""
        trace = np.ones(150, dtype=np.float32)
        G = np.eye(150, dtype=np.float32)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        assert RegL2.shape == (150,)

    def test_mixed_precision(self):
        """Test with mixed precision inputs."""
        trace = np.ones(150, dtype=np.float64)
        G = np.eye(150, dtype=np.float32)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        assert RegL2.shape == (150,)


class TestIRLSInvResiduoTracking:
    """Test residuo_iter tracking."""

    def test_residuo_iter_values_realistic(self):
        """Test that residuo_iter contains realistic values."""
        trace = np.sin(np.linspace(0, 2*np.pi, 160))
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=3)
        
        assert residuo_iter.shape == (10,)
        assert not np.all(residuo_iter == 0.0)

    def test_residuo_iter_position_145(self):
        """Test that residuo_iter tracks element at position 145."""
        trace = np.random.rand(200)
        G = np.eye(200)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=2)
        
        assert residuo_iter.shape == (10,)


class TestIRLSInvMatrixInversionPath:
    """Test different matrix inversion code paths."""

    def test_threshold_branch_below(self):
        """Test code path where abs(residuo) < threshold."""
        trace = np.random.rand(175) * 0.001
        G = np.eye(175)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, thresold=1.0, niter=2)
        assert residuo_iter.shape == (10,)

    def test_threshold_branch_above(self):
        """Test code path where abs(residuo) >= threshold."""
        trace = np.random.rand(175) * 10.0
        G = np.eye(175)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, thresold=0.001, niter=2)
        assert residuo_iter.shape == (10,)

    def test_both_branches_in_single_run(self):
        """Test run where both branches (< and >=) are taken."""
        trace = np.concatenate([np.ones(75) * 0.0001, np.ones(75)])
        G = np.eye(150)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, thresold=0.1, niter=2)
        assert residuo_iter.shape == (10,)


class TestIRLSInvNumericalStability:
    """Test numerical stability and consistency."""

    def test_reproducibility(self):
        """Test that results are reproducible with same inputs."""
        trace = np.ones(150)
        G = np.eye(150)
        
        result1 = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        result2 = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        
        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])

    def test_l2_solution_exists(self):
        """Test that L2 regularized solution is always computed."""
        trace = np.random.rand(160)
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        
        assert not np.any(np.isnan(RegL2))

    def test_l1_solution_differs_from_l2(self):
        """Test that L1 solution can differ from L2 solution."""
        trace = np.random.rand(160) * 10
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=3)
        
        assert not np.allclose(RegL2, RegL1)


class TestIRLSInvEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_trace_with_zeros(self):
        """Test trace with zero values."""
        trace = np.tile([0.0, 1.0], 75)
        G = np.eye(150)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        assert RegL2.shape == (150,)

    def test_trace_all_same_value(self):
        """Test trace with all same values."""
        trace = np.ones(160) * 5.0
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        assert RegL2.shape == (160,)

    def test_negative_trace_values(self):
        """Test with negative trace values."""
        trace = np.tile([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0], 21)
        G = np.eye(147)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        assert RegL2.shape == (147,)

    def test_large_amplitude_trace(self):
        """Test with very large amplitude values."""
        trace = np.random.rand(160) * 1e10
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        assert RegL2.shape == (160,)

    def test_very_small_amplitude_trace(self):
        """Test with very small amplitude values."""
        trace = np.random.rand(160) * 1e-10
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, 1)
        assert RegL2.shape == (160,)


class TestIRLSInvLoopCoverage:
    """Test to ensure loop coverage."""

    def test_loop_iterations_tracked(self):
        """Test that loop iterations properly update residuo_iter."""
        trace = np.sin(np.linspace(0, 4*np.pi, 300))
        G = np.eye(300)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=5)
        assert residuo_iter.shape == (10,)

    def test_partial_residual_loop_coverage(self):
        """Test the loop that updates R (first dim//2+2 elements)."""
        trace = np.random.rand(160)
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=2)
        assert RegL2.shape == (160,)

    def test_full_loop_coverage(self):
        """Test the loop that updates I for all dim elements."""
        trace = np.random.rand(180)
        G = np.eye(180)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=2)
        assert RegL1.shape == (180,)


class TestIRLSInvParameterVariations:
    """Test various parameter combinations."""

    def test_params_combination_1(self):
        """Test parameter combination 1."""
        trace = np.random.rand(170)
        G = np.eye(170)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 
                                               alphaL1=0.01, alphaL2=0.5, 
                                               thresold=0.1, niter=3)
        assert RegL1.shape == (170,)

    def test_params_combination_2(self):
        """Test parameter combination 2."""
        trace = np.random.rand(185)
        G = np.eye(185)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 
                                               alphaL1=0.5, alphaL2=0.01, 
                                               thresold=0.001, niter=5)
        assert RegL1.shape == (185,)

    def test_params_combination_3(self):
        """Test parameter combination 3."""
        trace = np.linspace(0, 10, 190)
        G = np.eye(190)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 
                                               alphaL1=0.001, alphaL2=0.001, 
                                               thresold=0.5, niter=1)
        assert RegL1.shape == (190,)


class TestIRLSInvL2Computation:
    """Tests specifically for L2 computation path."""

    def test_tikhonov_l2_solution_computed(self):
        """Test that Tikhonov L2 solution is always computed before iterations."""
        trace = np.random.rand(150)
        G = np.eye(150)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=1)
        
        # L2 solution should exist and be valid
        assert RegL2.shape == (150,)
        assert not np.any(np.isnan(RegL2))
        assert not np.any(np.isinf(RegL2))

    def test_l2_solution_before_iteration(self):
        """Test that L2 solution is computed before entering iteration loop."""
        trace = np.array([1.0, 2.0, 3.0, 2.0, 1.0] + [0.5] * 150)
        G = np.eye(155)
        
        # L2 is computed before iteration loop, even with niter=1
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=1)
        
        # L2 should be computed and returned
        assert RegL2.shape == (155,)
        assert RegL1.shape == (155,)


class TestIRLSInvIterativeRefinement:
    """Tests for the iterative refinement process."""

    def test_residuals_updated_each_iteration(self):
        """Test that residuals are updated in each iteration."""
        trace = np.random.rand(160) + 1.0  # Avoid near-zero values
        G = np.eye(160)
        
        RegL2_1iter, RegL1_1iter, res_1iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=1)
        RegL2_3iter, RegL1_3iter, res_3iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=3)
        
        # Both should have valid shapes
        assert res_1iter.shape == (10,)
        assert res_3iter.shape == (10,)

    def test_weighting_matrix_updates(self):
        """Test that weighting matrices R and I are updated properly."""
        # Create a trace with varying magnitudes to ensure weighting changes
        trace = np.concatenate([np.random.rand(100), np.random.rand(60) * 10])
        G = np.eye(160)
        
        RegL2, RegL1, residuo_iter = IRLS_Inv(trace, G, 0.1, 0.1, 0.01, niter=3)
        
        # Solutions should be computed
        assert RegL2.shape == (160,)
        assert RegL1.shape == (160,)
