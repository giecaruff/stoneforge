import pytest
import numpy as np
from stoneforge.pseudo_wells import referenced

def test_color_codes():
    lito, fluid = referenced.color_codes()
    assert isinstance(lito, dict)
    assert isinstance(fluid, dict)
    assert lito[22] == "darkgreen"
    assert fluid['brine'] == "navy"

def test_anadrill_siliciclastic_default(capsys):
    # Default is structure=False
    res = referenced.anadrill_siliciclastic()
    assert res is None
    captured = capsys.readouterr()
    assert "shale" in captured.out
    assert "clean_sandstone with gas" in captured.out

def test_anadrill_siliciclastic_with_structure():
    # structure as dict
    structure = {1: 5, 2: 10}
    res, units = referenced.anadrill_siliciclastic(structure=structure, step=0.5, top=100.0)
    assert isinstance(res, dict)
    assert isinstance(units, dict)
    
    # Check total samples = 5 + 10 = 15
    assert len(res['DEPTH']) == 15
    # Depth starts at top + step = 100.0 + 0.5 = 100.5, and increments by 0.5
    assert np.isclose(res['DEPTH'][0], 100.5)
    assert np.isclose(res['DEPTH'][-1], 100.0 + 15 * 0.5)
    
    # Units check
    assert units['DEPTH'] == 'm'
    assert units['GR'] == 'API'

def test_generate_invalid_structures():
    # Mismatched length in list/tuple
    with pytest.raises(ValueError) as excinfo:
        referenced.generate(structure=([1, 2], [10]))
    assert "facies and counts must have same length" in str(excinfo.value)
    
    # Invalid type/length
    with pytest.raises(TypeError) as excinfo:
        referenced.generate(structure=[1, 2, 3])
    assert "structure must be either dict or (facies_list, counts_list)" in str(excinfo.value)
    
    with pytest.raises(TypeError) as excinfo:
        referenced.generate(structure=123)
    assert "structure must be either dict or (facies_list, counts_list)" in str(excinfo.value)

def test_generate_list_of_2_elements():
    # structure is a list of length 2
    # This will enter: elif isinstance(structure, (list, tuple)) and len(structure) == 2:
    # and then hit the line: if type(structure) == type([]):
    # which redefines structure as (structure, _s) where _s = [1]*len(structure)
    # Wait, if structure is a list of length 2: [[1, 2], [10, 20]]
    # facies_seq = [1, 2], counts_seq = [10, 20]
    # structure is type list, so type(structure) == type([]) is True.
    # _n = len(structure) = 2. _s = [1, 1].
    # Then it does structure = (structure, _s) -> ([[1, 2], [10, 20]], [1, 1])
    # The generation uses facies_seq and counts_seq which are [1, 2] and [10, 20], total 30 samples.
    # Let's check if this executes successfully and returns the correct result.
    structure = [[1, 2], [10, 20]]
    res = referenced.generate(structure=structure)
    assert isinstance(res, dict)
    assert len(res['DEPTH']) == 30

def test_generate_noise_and_seed():
    structure = {1: 10}
    
    # random_state is False -> no noise
    res_no_noise = referenced.generate(structure=structure, random_state=False)
    
    # random_state is set -> deterministic noise
    res_noise_1 = referenced.generate(structure=structure, random_state=42)
    res_noise_2 = referenced.generate(structure=structure, random_state=42)
    res_noise_3 = referenced.generate(structure=structure, random_state=99)
    
    # Since noise is random, res_noise_1 should equal res_noise_2
    assert np.allclose(res_noise_1['GR'], res_noise_2['GR'])
    
    # With a different seed, it should differ
    assert not np.allclose(res_noise_1['GR'], res_noise_3['GR'])
    
    # Without noise, it should be different or equal to the mean block
    # We can check that noise actually modified the values compared to no-noise
    assert not np.allclose(res_no_noise['GR'], res_noise_1['GR'])
