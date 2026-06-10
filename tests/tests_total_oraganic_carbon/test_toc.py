import pytest
import numpy as np
from stoneforge.petrophysics import total_organic_carbon_content

def test_passey():
    dt = np.array([80.0, 90.0])
    rt = np.array([2.0, 3.0])
    dtbaseline = 55.0
    rtbaseline = 1.0
    lom = 10.6
    
    # Run passey
    toc = total_organic_carbon_content.passey(dt, rt, dtbaseline, rtbaseline, lom)
    
    assert len(toc) == 2
    # Verify clip behavior or expected formula
    # dlogrt = (rt - rtbaseline) + 0.02 * (dt - dtbaseline)
    # dlogrt_0 = (2 - 1) + 0.02 * (80 - 55) = 1.0 + 0.02 * 25 = 1.5
    # toc_0 = 1.5 * 10**(2.297 - 0.1688 * 10.6) = 1.5 * 10**(2.297 - 1.78928) = 1.5 * 10**(0.50772) = 1.5 * 3.219 = 4.8285
    assert np.isclose(toc[0], 4.8285, rtol=1e-3)
    assert np.all(toc >= 0.0)
    assert np.all(toc <= 100.0)

def test_calculate_toc():
    dt = np.array([80.0])
    rt = np.array([2.0])
    dtbaseline = 55.0
    rtbaseline = 1.0
    lom = 10.6
    
    toc = total_organic_carbon_content.calculate_toc(dt, rt, dtbaseline, rtbaseline, lom)
    assert len(toc) == 1
    assert np.isclose(toc[0], 4.8285, rtol=1e-3)
