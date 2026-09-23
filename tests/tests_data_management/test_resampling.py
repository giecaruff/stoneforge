import pytest
import pandas as pd
import numpy as np
from stoneforge.data_management._resampling import resampling

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'DEPTH': [100.0, 100.5, 101.0, 102.0],
        'GR': [40.0, 50.0, 60.0, 80.0],
        'FACIES': ['shale', 'shale', 'sandstone', 'sandstone']
    })

def test_resampling_nearest(sample_dataframe):
    # Test default resampling with nearest mode
    res = resampling(sample_dataframe, depth='DEPTH', step=1.0, top=100.0, bottom=102.0, mode='nearest')
    assert len(res) == 3 # 100.0, 101.0, 102.0
    assert np.allclose(res['DEPTH'], [100.0, 101.0, 102.0])
    
    # 100.0 is closest to 100.0 (GR=40.0)
    assert res.loc[0, 'GR'] == 40.0
    # 101.0 is closest to 101.0 (GR=60.0)
    assert res.loc[1, 'GR'] == 60.0
    # 102.0 is closest to 102.0 (GR=80.0)
    assert res.loc[2, 'GR'] == 80.0

def test_resampling_depth_window_and_swapping(sample_dataframe):
    # Test swapping top and bottom
    res = resampling(sample_dataframe, depth='DEPTH', step=1.0, top=102.0, bottom=100.0, mode='nearest')
    assert len(res) == 3
    assert np.allclose(res['DEPTH'], [100.0, 101.0, 102.0])
    
    # Test top-only bound (covers top is not None, bottom is None)
    res_top_only = resampling(sample_dataframe, depth='DEPTH', step=1.0, top=100.5, bottom=None, mode='nearest')
    assert np.isclose(res_top_only['DEPTH'].min(), 100.5)
    
    # Test bottom-only bound (covers top is None, bottom is not None)
    res_bot_only = resampling(sample_dataframe, depth='DEPTH', step=1.0, top=None, bottom=101.5, mode='nearest')
    assert np.isclose(res_bot_only['DEPTH'].max(), 102.0)
    
    # Test empty window error
    with pytest.raises(ValueError) as excinfo:
        resampling(sample_dataframe, depth='DEPTH', step=1.0, top=200.0, bottom=300.0)
    assert "No data in the specified depth range" in str(excinfo.value)

def test_resampling_bin_modes(sample_dataframe):
    # Test mode="mean"
    res_mean = resampling(sample_dataframe, depth='DEPTH', step=1.0, top=100.0, bottom=102.0, mode='mean')
    assert len(res_mean) == 3
    # For d=100.0, bin is [99.5, 100.5]. Samples are 100.0 (GR=40.0) and 100.5 (GR=50.0). Mean = 45.0
    assert np.isclose(res_mean.loc[0, 'GR'], 45.0)
    assert res_mean.loc[0, 'FACIES'] == 'shale'
    
    # Test mode="weighted_mean"
    res_wmean = resampling(sample_dataframe, depth='DEPTH', step=1.0, top=100.0, bottom=102.0, mode='weighted_mean')
    assert len(res_wmean) == 3
    # Since d=100.0:
    # 100.0 has dist = 0 -> weight = 1e6
    # 100.5 has dist = 0.5 -> weight = 2
    # So 100.0 dominates completely: GR should be very close to 40.0
    assert np.isclose(res_wmean.loc[0, 'GR'], 40.0, rtol=1e-4)

def test_resampling_least_squares(sample_dataframe):
    # Test mode="least_squares"
    res_lsq = resampling(sample_dataframe, depth='DEPTH', step=1.0, top=100.0, bottom=102.0, mode='least_squares')
    # For d=100.0: samples at 100.0 (GR=40.0) and 100.5 (GR=50.0). 
    # Linear fit: y = 20 * x - 1960. At x=100.0, y = 40.0.
    # Wait, polyval at center d=100.0 should give 40.0.
    assert np.isclose(res_lsq.loc[0, 'GR'], 40.0)
    
    # For d=102.0: only one sample (102.0) falls into the bin [201.5, 202.5]. 
    # Fallback in least_squares mode for len(values) == 1 should return values[0] = 80.0
    assert np.isclose(res_lsq.loc[2, 'GR'], 80.0)

def test_resampling_empty_bins_fallback():
    # Construct a dataset with a huge gap in depth
    df_gap = pd.DataFrame({
        'DEPTH': [100.0, 105.0],
        'GR': [40.0, 80.0],
        'FACIES': ['shale', 'sandstone']
    })
    # Resample with step=1.0. Bins at 101.0, 102.0, 103.0, 104.0 will be empty.
    res = resampling(df_gap, depth='DEPTH', step=1.0, top=100.0, bottom=105.0, mode='mean')
    assert len(res) == 6
    # Empty bins should fall back to nearest neighbor:
    # d=102.0: closest to 100.0 (GR=40.0)
    assert np.isclose(res.loc[2, 'GR'], 40.0)
    # d=103.0: closest to 105.0 (GR=80.0)
    assert np.isclose(res.loc[3, 'GR'], 80.0)

def test_resampling_unimplemented_mode(sample_dataframe):
    with pytest.raises(NotImplementedError) as excinfo:
        resampling(sample_dataframe, depth='DEPTH', step=1.0, top=100.0, bottom=102.0, mode='unimplemented')
    assert "Mode 'unimplemented' not implemented" in str(excinfo.value)
