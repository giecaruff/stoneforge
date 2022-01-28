
import sys
import os


if __package__:
    from ..petrophysics.water_saturation import water_saturation
else:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from petrophysics.water_saturation import water_saturation


def test_archie():
    assert round(water_saturation(rw=0.9, rt=20, phi=0.33,
                                a=0.62, m=2.15, n=2.0,
                                method="archie"),2) == 0.55

def test_simandoux():
    assert round(water_saturation(rw=0.015, rt=1.0, phi=0.11,
                                a=0.62, m=2.15, n=2.0,
                                method="simandoux", vsh=0.33,
                                            rsh=4.0),2) == 0.82

def test_indonesia(): 
    assert round(water_saturation(rw=17, rt=14.0, phi=0.23,
                                a=1.00, m=1.8, n=2.0,
                                method="indonesia", vsh=0.19,
                                rsh=4.0),2) == 0.22

def test_fertl(): 
    assert round(water_saturation(rw=0.015, rt=1.0, phi=0.11,
                                a=0.62, m=2.15, alpha=0.30,
                                method="fertl", vsh=0.33),2) == 0.63
