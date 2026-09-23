import pytest
import numpy as np
from stoneforge.inversion.viability_study.bayes import get_posterior_lithologies_prob


def test_get_posterior_basic():
    # Simple two-lithology case
    likelihoods = [0.2, 0.8]
    priors = [0.5, 0.5]

    post = get_posterior_lithologies_prob(likelihoods, priors)
    assert len(post) == 2
    assert pytest.approx(sum(post)) == 1.0
    # posterior should favor the higher likelihood
    assert post[1] > post[0]


def test_get_posterior_zero_likelihoods():
    # One lithology impossible under data
    likelihoods = [0.0, 0.5, 0.5]
    priors = [0.2, 0.4, 0.4]

    post = get_posterior_lithologies_prob(likelihoods, priors)
    assert len(post) == 3
    assert post[0] == 0.0
    assert pytest.approx(sum(post)) == 1.0


def test_get_posterior_extreme_priors():
    # Prior strongly favors first lithology
    likelihoods = [0.1, 0.9]
    priors = [0.99, 0.01]

    post = get_posterior_lithologies_prob(likelihoods, priors)
    assert len(post) == 2
    assert post[0] > post[1]
    assert pytest.approx(sum(post)) == 1.0


def test_get_posterior_numeric_stability():
    # Test with very small and very large numbers
    likelihoods = [1e-300, 1e-300, 1e-300]
    priors = [1e-300, 1e-300, 1e-300]

    post = get_posterior_lithologies_prob(likelihoods, priors)
    assert len(post) == 3
    # All equal -> should be uniform
    assert pytest.approx(post[0]) == post[1] == post[2]
    assert pytest.approx(sum(post)) == 1.0


def test_get_posterior_invalid_length():
    # Mismatched input lengths should raise via zip behavior (no explicit check in function)
    likelihoods = [0.1, 0.2]
    priors = [0.5]

    with pytest.raises(Exception):
        _ = get_posterior_lithologies_prob(likelihoods, priors)
