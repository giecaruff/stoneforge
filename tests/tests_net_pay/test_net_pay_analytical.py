# %%
import pytest
import numpy as np

if __package__:
    from ..reservoir.net_pay import net_pay_siliciclastic, cutoff
else:
    from reservoir.net_pay import net_pay_siliciclastic, cutoff

# -------------------------------------------------------------------------------------------------------------- #
# test functions

def test_net_pay_perfect_reservoir():
    vsh = [0.1, 0.2]
    phi = [0.25, 0.30]
    sw  = [0.1, 0.2]

    result = net_pay_siliciclastic(vsh, phi, sw)

    assert np.all(result['rock'] == 1.0)
    assert np.all(result['res'] == 1.0)
    assert np.all(result['pay'] == 1.0)
    