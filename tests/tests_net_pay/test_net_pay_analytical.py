# %%
import pytest
import numpy as np

if __package__:
    from ..reservoir.net_pay import net_pay_siliciclastic, cutoff
else:
    from reservoir.net_pay import net_pay_siliciclastic, cutoff

# -------------------------------------------------------------------------------------------------------------- #
# test functions

# 1. End-member correctness (perfect reservoir)
def test_net_pay_perfect_reservoir():
    vsh = [0.1, 0.2]
    phi = [0.25, 0.30]
    sw  = [0.1, 0.2]

    result = net_pay_siliciclastic(vsh, phi, sw)

    assert np.all(result['rock'] == 1.0)
    assert np.all(result['res'] == 1.0)
    assert np.all(result['pay'] == 1.0)

# 2. No reservoir rock (high shale)
def test_net_pay_no_rock():
    vsh = [0.5, 0.6]      # above threshold
    phi = [0.30, 0.35]
    sw  = [0.1, 0.2]

    result = net_pay_siliciclastic(vsh, phi, sw, fillzeros=True)

    assert np.all(result['rock'] == 0.0)
    assert np.all(result['res'] == 0.0)
    assert np.all(result['pay'] == 0.0)

# 3. Reservoir-quality porosity but invalid rock
def test_net_pay_porosity_cannot_override_rock():
    vsh = [0.6]           # fails rock
    phi = [0.35]          # good porosity
    sw  = [0.1]

    result = net_pay_siliciclastic(vsh, phi, sw, fillzeros=True)

    assert result['rock'][0] == 0.0
    assert result['res'][0] == 0.0
    assert result['pay'][0] == 0.0

# 4. Pay requires reservoir
def test_net_pay_sw_cannot_override_reservoir():
    vsh = [0.1]          # good rock
    phi = [0.05]         # fails reservoir
    sw  = [0.05]         # excellent Sw

    result = net_pay_siliciclastic(vsh, phi, sw, fillzeros=True)

    assert result['rock'][0] == 1.0
    assert result['res'][0] == 0.0
    assert result['pay'][0] == 0.0
    
# 5 (a). Threshold boundary behavior (exclusion)
def test_cutoff_threshold_exclusion():
    log = [0.3]

    result = cutoff(log, t=0.3, equal=False, fillzeros=True)

    assert result[0] == 0.0
    
# 5 (b). Threshold boundary behavior (inclusion)
def test_cutoff_threshold_inclusion():
    log = [0.3]

    result = cutoff(log, t=0.3, equal=True, fillzeros=True)

    assert result[0] == 1.0