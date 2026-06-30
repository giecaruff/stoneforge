import pytest
import numpy as np
from stoneforge.pseudo_wells import lithology_generator

def test_markov_chain():
    lito = [1, 1, 2, 2, 1, 2, np.nan]
    MC, states = lithology_generator.markov_chain(lito)
    
    assert set(states) == {1, 2}
    assert MC.shape == (2, 2)
    # Rows should sum to 1.0
    assert np.allclose(MC.sum(axis=1), [1.0, 1.0])

def test_simple_generator():
    mc = np.array([
        [0.8, 0.2],
        [0.4, 0.6]
    ])
    
    # 1. Without lithology_code
    res1 = lithology_generator.simple(mc, sampling=50, lithology_code=False, initial_state=0, seed_value=42)
    assert len(res1) == 50
    assert set(res1).issubset({0, 1})
    
    # 2. With lithology_code (initial_state must be one of the codes)
    codes = [10, 20]
    res2 = lithology_generator.simple(mc, sampling=50, lithology_code=codes, initial_state=10, seed_value=42)
    assert len(res2) == 50
    assert set(res2).issubset({10, 20})

def test_extended_generator():
    mc = np.array([
        [0.8, 0.2],
        [0.4, 0.6]
    ])
    codes = [10, 20]
    
    # 1. lithology_code=False, single_lithology=True
    res1 = lithology_generator.extended(mc, sampling=10, lithology_code=False, initial_state=0, single_lithology=True, seed_value=42)
    assert len(res1) == 10
    assert set(res1).issubset({0, 1})
    
    # 2. lithology_code=False, single_lithology=False
    res2 = lithology_generator.extended(mc, sampling=10, lithology_code=False, initial_state=0, single_lithology=False, seed_value=42)
    assert res2.shape == (256, 10)
    
    # 3. lithology_code=codes, single_lithology=True
    res3 = lithology_generator.extended(mc, sampling=10, lithology_code=codes, initial_state=0, single_lithology=True, seed_value=42)
    assert len(res3) == 10
    assert set(res3).issubset({10, 20})
    
    # 4. lithology_code=codes, single_lithology=False
    res4 = lithology_generator.extended(mc, sampling=10, lithology_code=codes, initial_state=0, single_lithology=False, seed_value=42)
    assert res4.shape == (256, 10)
