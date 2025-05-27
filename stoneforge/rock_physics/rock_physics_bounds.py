import numpy as np
from typing import Annotated

def reuss(
    f: Annotated[np.array, "Fractions (proportions) of each mineral"],
    m: Annotated[np.array, "Elastic modulus of each mineral"])-> np.array:
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
    return 1 / np.sum(f/m, axis=0)


def voigt(
    f: Annotated[np.array, "Fractions (proportions) of each mineral"],
    m: Annotated[np.array, "Elastic modulus of each mineral"])-> np.array:
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
    return np.sum(f*m, axis=0)


def hill(
    f: Annotated[np.array, "Fractions (proportions) of each mineral"],
    m: Annotated[np.array, "Elastic modulus of each mineral"])-> np.array:
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
    return (r+v)/2
