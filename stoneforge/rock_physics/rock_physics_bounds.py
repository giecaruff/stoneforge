import numpy as np


def reuss(f, m):
    """Calculate elastic modulus of the effective mineral using the
    Reuss bound.

    Parameters
    ----------
    f : array_like
        Fractions (proportions) of each mineral.
    m : array_like
        Elastic modulus of each mineral.

    Returns
    -------
    r: array_like
        Reuss bound.
    
    """
    r = 1 / np.sum(f/m)
    return r


def voigt(f, m):
    """Calculate elastic modulus of the effective mineral using the
    Reuss bound.

    Parameters
    ----------
    f : array_like
        Fractions (proportions) of each mineral.
    m : array_like
        Elastic modulus of each mineral.

    Returns
    -------
    v: array_like
        Voigt bound.
    
    """
    v = np.sum(f*m)
    return v


def hill(f, m):
    """Calculate elastic modulus of the effective mineral using the
    Hill average.

    Parameters
    ----------
    f : array_like
        Fractions (proportions) of each mineral.
    m : array_like
        Elastic modulus of each mineral.

    Returns
    -------
    v: array_like
        Hill average.
    
    """
    r = 1 / np.sum(f/m)
    v = np.sum(f*m)
    h = (r+v)/2
    return h
