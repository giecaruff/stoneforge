import sys
import os


if __package__:
    from ..fluid_substitution import gassmann
else:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from fluid_substitution import gassmann


def test_kdry():
    assert round(gassmann(phi=0.2, ks=36, ksatA=17.6, kfluidA=3.013,
                          method="kdry"), 1) == 12.


def test_ksat():
    assert round(gassmann(phi=0.2, ks=36, kdry=12, kfluidB=3.013,
                          method="ksat"), 1) == 17.6


def test_gassmann_subs():
    assert round(gassmann(phi=0.2, ks=36, ksatA=12.29, kfluidA=0.133,
                 kfluidB=3.013, method="gassmann_subs"), 1) == 17.6
