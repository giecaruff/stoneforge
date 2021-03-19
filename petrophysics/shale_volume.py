import numpy as np
import pytest


def gammarayindex(gr, grmin, grmax):
    """Calculates the gamma ray index.

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
    igr : array_like
        The gamma ray index varying between 0.0 (clean sand) and 1.0 (shale).
    
    """

    igr = (gr - grmin) / (grmax - grmin)
    igr = np.clip(igr, 0.0, 1.0)

    return igr


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

    vshale = gammarayindex(gr, grmin, grmax)

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

    igr = gammarayindex(gr, grmin, grmax)
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

    igr = gammarayindex(gr, grmin, grmax)
    vshale = 0.083 * (2 ** (3.7 * igr) - 1)

    return vshale


_vshale_methods = {
    "linear": vshale_linear,
    "larionov": vshale_larionov,
    "larionov_terciary": vshale_larionov_terciary,
}


def vshale(gr, grmin, grmax, method=None):
    """Compute the shale volume from gamma ray log.

    This is a façade for the methods:
        - vshale_linear
        - vshale_larionov
        - vshale_larionov_terciary

    Parameters
    ----------
    gr : array_like
        Gamma Ray log.
    grmin : int, float
        Clean sand GR value.
    grmax : int, float
        Shale/clay value.
    method : str, optional
        Name of the method to be used.  Should be one of
            - 'linear'
            - 'larionov'
            - 'larionov_terciary'
        If not given, default method is 'linear'

    Returns
    -------
    vshale : array_like
        Shale Volume for the aimed interval using the Larionov method.
    """
    if method is None:
        method = "linear"
    
    fun = _vshale_methods[method]

    return fun(gr, grmin, grmax)
