from cgi import test
import sys
import os


if __package__:
    from ..rock_physics.fluid_substitution import gassmann
    from ..rock_physics.fluid_substitution import mavko
else:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from rock_physics.fluid_substitution import gassmann
    from rock_physics.fluid_substitution import mavko


def test_kdry():
    assert round(gassmann(phi=0.2, ks=36, ksatA=17.6, kfluidA=3.013,
                          method="kdry"), 1) == 12.


def test_ksat():
    assert round(gassmann(phi=0.2, ks=36, kdry=12, kfluidB=3.013,
                          method="ksat"), 1) == 17.6


def test_gassmann_subs():
    assert round(gassmann(phi=0.2, ks=36, ksatA=12.29, kfluidA=0.133,
                          kfluidB=3.013, method="gassmann_subs"), 1) == 17.6


def test_mdry():
    assert round(mavko(phi=0.2, ms=97, msatA=22.5, kfluidA=3.013,
                 method="mdry"), 1) == 12.


def test_msat():
    assert round(mavko(phi=0.2, ms=97, mdry=12., kfluidB=3.013,
                       method="msat"), 1) == 22.5


def test_mavko_subs():
    assert round(mavko(phi=0.2, ms=97, msatA=22.5, kfluidA=2.25,
                       kfluidB=3.013, method="mavko_subs"), 1) == 24.8
