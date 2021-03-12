import numpy as np
import pytest
def water_saturation(rw,rt,phi,a,m,n):
    """Estimate the Water Saturation from Archie method.
    Parameters
    ----------
    rw : array_like
        Water resistivity.
    rt : array_like
        True resistivity.    
    phi : array_like
        Porosity (must be effective).         
    a : int, float
        Tortuosity factor.
    m : int, float
        Cementation exponent.
    n : int, float
        Saturation exponent.

    Returns
    -------

    sw : array_like
        Porosity from sonic.
    """

    sw = ((a*rw)/(phi**m*rt))**(1./n)

    return sw