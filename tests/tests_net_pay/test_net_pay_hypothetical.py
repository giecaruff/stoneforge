# %%
import numpy as np
from hypothesis import given, strategies as st, assume

if __package__:
    from ..reservoir.net_pay import net_pay_siliciclastic, cutoff
else:
    from reservoir.net_pay import net_pay_siliciclastic, cutoff

# -------------------------------------------------------------------------------------------------------------- #
# test functions

# 1. Fundamental logical hierarchy invariant
@given(
    vsh=st.floats(min_value=0.0, max_value=1.0),
    phi=st.floats(min_value=0.0, max_value=0.4),
    sw=st.floats(min_value=0.0, max_value=1.0),
)
def test_pay_implies_reservoir_and_rock(vsh, phi, sw):
    result = net_pay_siliciclastic([vsh], [phi], [sw], fillzeros=True)

    pay  = result['pay'][0]
    res  = result['res'][0]
    rock = result['rock'][0]

    if pay == 1.0:
        assert res == 1.0
        assert rock == 1.0
        
        
# 2. Reservoir requires rock
@given(
    vsh=st.floats(min_value=0.31, max_value=1.0),  # guaranteed bad rock
    phi=st.floats(min_value=0.0, max_value=0.4),
    sw=st.floats(min_value=0.0, max_value=1.0),
)
def test_no_rock_means_no_reservoir(vsh, phi, sw):
    result = net_pay_siliciclastic([vsh], [phi], [sw], fillzeros=True)

    assert result['rock'][0] == 0.0
    assert result['res'][0]  == 0.0
    assert result['pay'][0]  == 0.0
    
    
# 3. Monotonicity with shale volume
@given(
    phi=st.floats(min_value=0.15, max_value=0.35),
    sw=st.floats(min_value=0.0, max_value=0.5),
    vsh1=st.floats(min_value=0.0, max_value=0.2),
    vsh2=st.floats(min_value=0.4, max_value=1.0),
)
def test_increasing_vsh_cannot_create_rock(phi, sw, vsh1, vsh2):
    assume(vsh2 > vsh1)

    r1 = net_pay_siliciclastic([vsh1], [phi], [sw], fillzeros=True)
    r2 = net_pay_siliciclastic([vsh2], [phi], [sw], fillzeros=True)

    assert r2['rock'][0] <= r1['rock'][0]
    
    
# 4. Monotonicity with porosity
@given(
    vsh=st.floats(min_value=0.0, max_value=0.3),
    sw=st.floats(min_value=0.0, max_value=0.5),
    phi1=st.floats(min_value=0.0, max_value=0.08),
    phi2=st.floats(min_value=0.15, max_value=0.35),
)
def test_increasing_phi_cannot_reduce_reservoir(vsh, sw, phi1, phi2):
    assume(phi2 > phi1)

    r1 = net_pay_siliciclastic([vsh], [phi1], [sw], fillzeros=True)
    r2 = net_pay_siliciclastic([vsh], [phi2], [sw], fillzeros=True)

    assert r2['res'][0] >= r1['res'][0]
    
    
# 5. Monotonicity with water saturation (very slow)
@given(
    vsh=st.floats(min_value=0.0, max_value=0.3),
    phi=st.floats(min_value=0.15, max_value=0.35),
    sw1=st.floats(min_value=0.0, max_value=0.3),
    sw2=st.floats(min_value=0.6, max_value=1.0),
)
def test_increasing_sw_cannot_improve_pay(vsh, phi, sw1, sw2):
    assume(sw2 > sw1)

    r1 = net_pay_siliciclastic([vsh], [phi], [sw1], fillzeros=True)
    r2 = net_pay_siliciclastic([vsh], [phi], [sw2], fillzeros=True)

    assert r2['pay'][0] <= r1['pay'][0]
    
    
# 6. Output domain invariant
@given(
    vsh=st.lists(st.floats(min_value=0, max_value=1), min_size=10, max_size=50),
    phi=st.lists(st.floats(min_value=0, max_value=0.4), min_size=10, max_size=50),
    sw=st.lists(st.floats(min_value=0, max_value=1), min_size=10, max_size=50),
)
def test_output_domain_binary(vsh, phi, sw):
    assume(len(vsh) == len(phi) == len(sw))

    result = net_pay_siliciclastic(vsh, phi, sw, fillzeros=True)

    for key in ['rock', 'res', 'pay']:
        values = result[key]
        assert np.all((values == 0.0) | (values == 1.0))