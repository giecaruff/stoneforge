# %%
import pytest
import numpy as np

if __package__:
    from ..reservoir.net_pay import net_pay_siliciclastic, cutoff
else:
    from reservoir.net_pay import net_pay_siliciclastic, cutoff

# -------------------------------------------------------------------------------------------------------------- #
# test functions

# 1. Vectorization + NaN propagation
def test_net_pay_vectorized_nan_behavior():
    vsh = [0.1, np.nan, 0.2]
    phi = [0.25, 0.30, np.nan]
    sw  = [0.1, 0.2, 0.3]

    result = net_pay_siliciclastic(vsh, phi, sw)

    assert np.isnan(result['rock'][1])
    assert np.isnan(result['res'][2])

# 2. Physical bounds invariant
def test_net_pay_output_domain():
    vsh = np.linspace(0, 1, 100)
    phi = np.linspace(0, 0.4, 100)
    sw  = np.linspace(0, 1, 100)

    result = net_pay_siliciclastic(vsh, phi, sw)

    for key in ['rock', 'res', 'pay']:
        valid = np.isnan(result[key]) | (result[key] == 1.0)
        assert np.all(valid)