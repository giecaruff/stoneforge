import sys
import os
import pytest

if __package__:
    from ..rock_physics.gem import gem
else:
    from stoneforge.rock_physics.gem import gem

# Just bounds testing
def test_soft_sand():
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="soft_sand",
               p=27*10**6)[0] < 10**15
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="soft_sand",
               p=27*10**6)[1] < 10**15
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="soft_sand",
               p=27*10**6)[0] > 0.0
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="soft_sand",
               p=27*10**6)[1] > 0.0


def test_stiff_sand():
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="stiff_sand",
               p=27*10**6)[0] < 10**15
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="stiff_sand",
               p=27*10**6)[1] < 10**15
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="stiff_sand",
               p=27*10**6)[0] > 0.0
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="stiff_sand",
               p=27*10**6)[1] > 0.0


def test_contact_cement():
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="contact_cement",
               kc=36*10**9, gc=45*10**9)[0] < 10**15
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="contact_cement",
               kc=36*10**9, gc=45*10**9)[1] < 10**15
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="contact_cement",
               kc=36*10**9, gc=45*10**9)[0] > 0.0
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="contact_cement",
               kc=36*10**9, gc=45*10**9)[1] > 0.0

def test_constant_cement():
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="constant_cement",
               kc=36*10**9, gc=45*10**9, phib=0.34)[0] < 10**15
    assert gem(36*10**9, 45*10**9, 0.0, 0.4, 5, method="constant_cement",
               kc=36*10**9, gc=45*10**9, phib=0.34)[1] < 10**15
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="constant_cement",
               kc=36*10**9, gc=45*10**9, phib=0.34)[0] > 0.0
    assert gem(36*10**9, 45*10**9, 0.4, 0.4, 5, method="constant_cement",
               kc=36*10**9, gc=45*10**9, phib=0.34)[1] > 0.0
