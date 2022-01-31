# %%
import sys
import os


if __package__:
    from ..petrophysics.porosity import porosity
else:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from petrophysics.porosity import porosity


def test_density_porosity():
    assert round(porosity(rhob=2.70, rhom=2.65, rhof=1.10,
                                method="density_porosity"),2) == 0.55