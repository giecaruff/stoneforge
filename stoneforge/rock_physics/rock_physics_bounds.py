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
    r = 1 / np.sum(f/m, axis=0)
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
    v = np.sum(f*m, axis=0)
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
    r = 1 / np.sum(f/m, axis=0)
    v = np.sum(f*m, axis=0)
    h = (r+v)/2
    return h
