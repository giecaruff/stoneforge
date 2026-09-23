import pytest
import numpy as np
from stoneforge.reservoir.toc_v import TOCProcessor

def test_passey():
    dt = np.array([80.0, 90.0])
    logrt = np.array([2.0, 3.0])
    gr = np.array([50.0, 60.0])
    cali = np.array([10.0, 12.0])
    measured_cot = np.array([5.0, 6.0])
    measured_cot_depth = np.array([1000.0, 1003.0])
    depth = np.array([1000.0, 1003.0])
    dt_base = 90.0
    res_base = 1.0
    lom_val = 10.0
    window_value = 100

    processor = TOCProcessor(
            depth = depth,
            dt = dt,
            gr = gr,
            logrt = logrt,
            cali = cali,
            measured_cot = measured_cot,
            measured_cot_depth = measured_cot_depth,
            window_value = window_value,
            lom=lom_val,
            dt_baseline=dt_base,
            logrt_baseline=res_base,
            skip=1
        )

    result = processor.run()
    #assert np.isclose(result[0], 4.8285, rtol=1e-3)
    assert np.all(result['calculated_cot_pct'] >= 0.0)
    assert np.all(result['calculated_cot_pct'] <= 10.0)
