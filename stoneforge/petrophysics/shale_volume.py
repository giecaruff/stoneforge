import numpy as np
import numpy.typing as npt


def gammarayindex(gr: npt.ArrayLike, grmin: float, grmax: float) -> np.ndarray:
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

    if grmin == grmax:
        msg = "Division by zero. The value of grmin is equal to the value of grmax."
        raise ZeroDivisionError(msg)

    igr = (gr - grmin) / (grmax - grmin)
    igr = np.clip(igr, 0.0, 1.0)

    return igr


def vshale_linear(gr: npt.ArrayLike, grmin: float, grmax: float) -> np.ndarray:
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


def vshale_larionov_old(gr: npt.ArrayLike, grmin: float, grmax: float) -> np.ndarray:
    """Estimate the shale volume from the Larionov model for old rocks.

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


def vshale_larionov(gr: npt.ArrayLike, grmin: float, grmax: float) -> np.ndarray:
    """Estimate the shale volume from the Larionov model for young rocks.

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

def vshale_clavier(gr: npt.ArrayLike, grmin: float, grmax: float):
    """Estimate the shale volume from the Clavier model.

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
        Shale Volume for the aimed interval using the Clavier method.
    """
    
    igr = gammarayindex(gr, grmin, grmax)
    vshale = 1.7 - np.sqrt(3.38 - (igr + 0.7) ** 2)

    return vshale

def vshale_stieber(gr: npt.ArrayLike, grmin: float, grmax: float):
    """Estimate the shale volume from the Stieber model.

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
        Shale Volume for the aimed interval using the Stieber method.
    """
    
    igr = gammarayindex(gr, grmin, grmax)
    vshale = igr / (3 - 2 * igr)

    return vshale


_vshale_methods = {
    "linear": vshale_linear,
    "larionov": vshale_larionov,
    "larionov_old": vshale_larionov_old,
    "clavier": vshale_clavier,
    "stieber": vshale_stieber,
}


def vshale(gr: npt.ArrayLike, grmin: float, grmax: float, method: str = None) -> np.ndarray:
    """Compute the shale volume from gamma ray log.

    This is a façade for the methods:
        - vshale_linear
        - vshale_larionov
        - vshale_larionov_old
        - vshale_clavier
        - vshale_stieber

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
            - 'larionov_old'
            - 'clavier'
            - 'stieber'
        If not given, default method is 'linear'

    Returns
    -------
    vshale : array_like
        Shale Volume for the aimed interval using defined method.
    """
    if method is None:
        method = "linear"

    if method not in _vshale_methods:
        msg = f"Method not found: {method}"
        raise ValueError(msg)
    
    fun = _vshale_methods[method]

    return fun(gr, grmin, grmax)
