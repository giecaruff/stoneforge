import pytest
import pandas as pd
import numpy as np
import matplotlib
# Use headless backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch
from stoneforge.vis import img

@pytest.fixture
def mock_well_data():
    return pd.DataFrame({
        'DEPTH': [100.0, 101.0, 102.0, 103.0],
        'GR': [45.0, 50.0, 80.0, 75.0],
        'RES': [10.0, 12.0, 3.0, 5.0],
        'LITH': [22, 22, 25, 27]
    })

@patch("matplotlib.pyplot.show")
def test_fastplot(mock_show, mock_well_data):
    curves = ['GR', 'RES']
    colors = ['red', 'blue']
    units = ['API', 'ohm.m']
    
    # Run fastplot
    img.fastplot(mock_well_data, 'DEPTH', curves, colors, units, d_unit='m', size=(6, 5))
    
    # Verify plot was called and show was called once
    mock_show.assert_called_once()
    plt.close('all')

def test_plito():
    lithology = [22, 22, 25, 25]
    depth = [10.0, 11.0, 12.0, 13.0]
    colors = {
        22: 'green',
        25: 'gray'
    }
    
    img.plito(lithology, depth, colors, linewidth=1.5)
    plt.close('all')

@patch("matplotlib.pyplot.show")
def test_plotwell_class(mock_show, mock_well_data):
    curves = ['GR', 'RES']
    colors = ['red', 'blue']
    units = ['API', 'ohm.m']
    
    pw = img.plotwell(mock_well_data, 'DEPTH', curves, colors, units, d_unit='m', size=(10, 8))
    
    # Add facies track
    facies_colors = {
        22: "darkgreen",
        25: "grey",
        27: "orange"
    }
    pw.facies('LITH', facies_colors, linewidth=0.5)
    
    # Call show
    pw.show()
    mock_show.assert_called_once()
    plt.close('all')
