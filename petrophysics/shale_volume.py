import numpy as np
import pytest
def vshale_linear(gr, grmin, grmax):
    """Estimate the shale volume from the linear model.

    Parameters
    ----------
    gr : array_like
        Gamma Ray log.
    grmin : int, float
        Clean sand GR value.
    grmax : int, float
        Shale/clay value. 

    Returns
    -------
    vshale : array_like
        Shale Volume for the aimed interval using the Linear method.
    """ 

    vshale = (gr - grmin) / (grmax - grmin)

    return vshale

def vshale_larionov(gr, grmin, grmax):
    """Estimate the shale volume from the Larionov model.

    Parameters
    ----------
    gr : array_like
        Gamma Ray log.
    grmin : int, float
        Clean sand GR value.
    grmax : int, float
        Shale/clay value.  

    Returns
    -------
    vshale : array_like
        Shale Volume for the aimed interval using the Larionov method.
    """

    igr = (gr - grmin) / (grmax - grmin)
    vshale = 0.33 * (2. ** (2. * igr) - 1)

    return vshale

def vshale_larionov_terciary(gr, grmin, grmax):
    """Estimate the shale volume from the Larionov model for Tertiary rocks.

    Parameters
    ----------
    gr : array_like
        Gamma Ray log.
    grmin : int, float
        Clean sand GR value.
    grmax : int, float
        Shale/clay value.
         
    Returns
    -------
    vshale : array_like
        Shale Volume for the aimed interval using the Larionov method.
    """

    igr = (gr - grmin) / (grmax - grmin)
    vshale = 0.083 * (2 ** (3.7 * igr) - 1)

    return vshale    
