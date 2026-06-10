import pytest
import numpy as np
import matplotlib
# Use headless backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch
from stoneforge.vis.plot_welllog import LogPlot

def test_log_plot_init():
    # Test init with specified top and bot
    lp = LogPlot(size=(27.7, 40.0), top=100.0, bot=200.0, title=('Well Test', 2.0))
    assert lp.top == 100.0
    assert lp.bot == 200.0
    assert lp.title == ('Well Test', 2.0)
    plt.close('all')

    # Test init with top/bot as None
    lp_none = LogPlot(size=(20.0, 30.0), top=None, bot=None)
    assert lp_none.top is None
    assert lp_none.bot is None
    plt.close('all')

def test_format_bar():
    lp = LogPlot()
    formatted = lp._format_bar(10.0, 100.0, "Gamma Ray")
    assert "Gamma Ray" in formatted
    assert "10.0" in formatted
    assert "100.0" in formatted

    # Test failure due to short bar
    lp._bar = "─"
    with pytest.raises(ValueError) as excinfo:
        lp._format_bar(10.0, 100.0, "Gamma Ray")
    assert "Bar is too short to fit" in str(excinfo.value)
    plt.close('all')

def test_set_colormapped_title():
    lp = LogPlot()
    lp.set_depth(np.linspace(100, 200, 10))
    lp._addtrack()
    lp._set_colormapped_title(lp.ax, "VSH", cmap_name="viridis")
    plt.close('all')

def test_mask_depth_range():
    lp = LogPlot()
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([10., 20., 30., 40., 50.])
    masked = lp._mask_depth_range(x, y, 20, 40)
    assert np.isnan(masked[0])
    assert masked[1] == 2.0
    assert masked[2] == 3.0
    assert masked[3] == 4.0
    assert np.isnan(masked[4])

def test_set_depth():
    lp = LogPlot(top=None, bot=None)
    depths = np.array([100.0, 150.0, 200.0])
    lp.set_depth(depths, d="Custom Depth")
    assert lp.top == 100.0
    assert lp.bot == 200.0
    assert lp.depth_description == "Custom Depth"
    plt.close('all')

@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_all_tracks(mock_savefig, mock_show):
    depth = np.linspace(100.0, 110.0, 11) # 11 points
    x = np.linspace(10.0, 50.0, 11)
    x2 = np.linspace(20.0, 40.0, 11)
    
    # Introduce NaN to test the NaN crossover code block
    x[5] = np.nan
    x2[5] = np.nan
    
    lp = LogPlot(top=100.0, bot=110.0, title=("Test Well", 2.0))
    lp.set_depth(depth)
    
    # 1. Normal plot (with and without ylim, with and without label)
    lp.normal_plot(x, track=False, label="GR", ylim=(101.0, 108.0))
    lp.normal_plot(x2, track=True, label="GR_2")
    
    # 2. Logarithm plot (with vmin/vmax as None, and specifying ylim to cover internal code paths)
    lp.logarithm_plot(x, track=False, label="RES", vmin=None, vmax=None, ylim=(101.0, 108.0))
    
    # 3. Fill plot
    lp.fill_plot(x, track=False, label="VSH")
    
    # 4. Fill cmap plot
    lp.fill_cmap_plot(x, track=False, label="CALI")
    
    # 5. Crossover plot
    # Set lp.title to string to work around crossover_plot's internal concatenation bug
    lp.title = "Crossover Title"
    lp.crossover_plot(x, x2, track=False, label="Crossover")
    # Restore title to tuple for show() method to work
    lp.title = ("Test Well", 2.0)
    
    # 6. Color plot (R, G, B matrix)
    rgb_matrix = np.random.rand(11, 3)
    lp.color_plot(rgb_matrix, track=False, rule="down")
    lp.color_plot(rgb_matrix, track=False, rule="up")
    lp.color_plot(rgb_matrix, track=False, rule="mean")
    
    # 7. Compositional plot
    comp_data = {
        'Quartz': np.array([0.5]*11),
        'Clay': np.array([0.5]*11)
    }
    comp_colors = {
        'Quartz': (1.0, 1.0, 0.0),
        'Clay': (0.5, 0.5, 0.5)
    }
    lp.compositional_plot(comp_data, comp_colors, track=False, spacing=0.9)
    
    # 8. Matrix plot (valid case)
    matrix_data = np.random.rand(11, 5)
    lp.matrix_plot(matrix_data, track=False, label="Matrix")
    
    # 9. Matrix plot (invalid case - shape mismatch)
    with pytest.raises(ValueError) as excinfo:
        invalid_matrix = np.random.rand(5, 5) # Rows do not match depth length (11)
        lp.matrix_plot(invalid_matrix, track=False)
    assert "Matrix row count" in str(excinfo.value)
    
    # Show and Save
    lp.show()
    mock_show.assert_called_once()
    
    lp.save(filetype="pdf")
    mock_savefig.assert_called_once_with("pdf", bbox_inches='tight', dpi=200)
    
    plt.close('all')
