# %%
import sys
import os
import pytest
import parameters_ws
#from parameters_ws import Parameters

if __package__:
    from ..petrophysics.water_saturation import water_saturation
else:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from petrophysics.water_saturation import water_saturation

# -------------------------------------------------------------------------------------------------------------- #
# test functions

archie_values = parameters_ws.Parameters.sorted_values(parameters_ws.Parameters.config_archie)

@pytest.mark.parametrize(archie_values[1], archie_values[0])
def test_archie(rw, rt, phi, a, m, n):
    ws = water_saturation(rw = rw, rt = rt, phi = phi, a = a,
                        m = m, n = n, method = "archie")
    assert any(ws >= 0) and any(ws <= 1)

# -------------------------------------------------------------------------------------------------------------- #
"""
simandoux_values = Parameters.sorted_values(Parameters.config_simandoux)

@pytest.mark.parametrize(simandoux_values[1], simandoux_values[0])
def test_simandoux(rw, rt, phi, a, m, n, vsh, rsh):
    ws = water_saturation(rw = rw, rt = rt, phi = phi, a = a,
                        m = m, n = n, vsh = vsh, rsh = rsh, method = "simandoux")
    assert ws >= 0 and ws <= 1

# -------------------------------------------------------------------------------------------------------------- #

indonesia_values = Parameters.sorted_values(Parameters.config_indonesia)

@pytest.mark.parametrize(indonesia_values[1], indonesia_values[0])
def test_indonesia(rw, rt, phi, a, m, n, vsh, rsh):
    ws = water_saturation(rw = rw, rt = rt, phi = phi, a = a,
                        m = m, n = n, vsh = vsh, rsh = rsh, method = "indonesia")
    assert ws >= 0 and ws <= 1

# -------------------------------------------------------------------------------------------------------------- #

fertl_values = Parameters.sorted_values(Parameters.config_fertl)

@pytest.mark.parametrize(fertl_values[1], fertl_values[0])
def test_fertl(rw, rt, phi, a, m, alpha, vsh):
    ws = water_saturation(rw = rw, rt = rt, phi = phi, a = a,
                        m = m, alpha = alpha, vsh = vsh, method = "fertl")
    assert ws >= 0 and ws <= 1

"""