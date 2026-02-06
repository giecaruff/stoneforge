# -*- coding: utf-8 -*-

import numpy as np
from typing import Annotated
#from stoneforge.petrophysics.helpers import correct_petrophysic_estimation_range
from .helpers import correct_petrophysic_estimation_range

def gammarayindex(
    gr: Annotated[np.array, "Gamma Ray log"],
    grmin: Annotated[float, "Clean GR value"],
    grmax: Annotated[float, "hale/clay value"]) -> np.array:
    """Calculates the gamma ray index :footcite:t:`schon1998physical`.

    Parameters
    ----------
    gr : array_like
        Gamma Ray log.
    grmin : float
        Clean sand GR value.
    grmax : float
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


def vshale_linear(
    gr: Annotated[np.array, "Gamma Ray log"],
    grmin: Annotated[float, "Clean GR value"],
    grmax: Annotated[float, "hale/clay value"]) -> np.array:
    """Estimate the shale volume from the linear model :footcite:t:`schon1998physical`.

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
    vshale = correct_petrophysic_estimation_range(vshale)
  
    return vshale


def vshale_larionov_old(
    gr: Annotated[np.array, "Gamma Ray log"],
    grmin: Annotated[float, "Clean GR value"],
    grmax: Annotated[float, "hale/clay value"]) -> np.array:
    """Estimate the shale volume from the Larionov model for old rocks :footcite:t:`larionov1969borehole, schon1998physical`.

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
    vshale = correct_petrophysic_estimation_range(vshale)
    return vshale


def vshale_larionov(
    gr: Annotated[np.array, "Gamma Ray log"],
    grmin: Annotated[float, "Clean GR value"],
    grmax: Annotated[float, "hale/clay value"]) -> np.array:
    """Estimate the shale volume from the Larionov model for young rocks :footcite:t:`larionov1969borehole, schon1998physical`.

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
    vshale = 0.083 * (2 ** (3.71 * igr) - 1)
    vshale = correct_petrophysic_estimation_range(vshale)

    return vshale


def vshale_clavier(
    gr: Annotated[np.array, "Gamma Ray log"],
    grmin: Annotated[float, "Clean GR value"],
    grmax: Annotated[float, "hale/clay value"]) -> np.array:
    """Estimate the shale volume from the Clavier model :footcite:t:`clavier1971, schon1998physical`.

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
    vshale = correct_petrophysic_estimation_range(vshale)

    return vshale


def vshale_stieber(
    gr: Annotated[np.array, "Gamma Ray log"],
    grmin: Annotated[float, "Clean GR value"],
    grmax: Annotated[float, "hale/clay value"]) -> np.array:
    """Estimate the shale volume from the Stieber model :footcite:t:`stieber1970, schon1998physical`.

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
    vshale = correct_petrophysic_estimation_range(vshale)

    return vshale


def vshale_neu_den(
    nphi: Annotated[np.array, "Neutron porosity log"],
    rhob: Annotated[np.array, "Bulk density log"],
    clean_n: Annotated[float, "Clean neutron point"] = -0.15,
    clean_d: Annotated[float, "Clean density point"] = 2.65,
    fluid_n: Annotated[float, "fluid neutron point"] = 1.00,
    fluid_d: Annotated[float, "fluid density point"] = 1.10,
    clay_n: Annotated[float, "Clay neutron point"] = 0.47,
    clay_d: Annotated[float, "Clay density point"] = 2.71) -> np.array:
    """Estimates the shale volume from neutron and density logs method (three points method) :footcite:t:`passeybhuyan1994`.

    Parameters
    ----------
    nphi : array_like
        Neutron porosity log.
    rhob : array_like
        Bulk density log.
    clean_n : -0.15, float
        Neutron porosity value from clean portion (base quartz).
    clean_d : 2.65, float
        Bulk density value from clean portion (base quartz).
    fluid_n : 1.00, float
        Neutron porosity value from fluid (base brine).
    fluid_d : 1.10, float
        Bulk density value from fluid (base brine).
    clay_n : 0.47, float
        Neutron porosity value from clay point (base standard shale).
    clay_d : 2.71, float
        Bulk density value from clay point (base standard shale).

    Returns
    -------
    vshale : array_like
        Shale volume from neutron and density logs method.

    """
    x1 = (fluid_d - clean_d) * (nphi - clean_n)
    x2 = (rhob - clean_d) * (fluid_n - clean_n)
    x3 = (fluid_d - clean_d) * (clay_n - clean_n)
    x4 = (clay_d - clean_d) * (fluid_n - clean_n)
    vshale = (x1-x2) / (x3-x4)
    #vshale = correct_petrophysic_estimation_range(vshale)
    return vshale

def vshale_nrm(
    phit: Annotated[np.array, "Total porosity log"],
    phie: Annotated[np.array, "Effective porosity log"]) -> np.array:
    """Estimate the shale volume from NMR curves :footcite:t:`passeybhuyan1994`.

    Parameters
    ----------
    phit : array_like
        Total porosity log from nmr.
    phie : int, float
        Effective porosity log from nmr.
         
    Returns
    -------
    vshale : array_like
        Shale Volume for the aimed interval using NMR curves.

    """
    cbw = phie - phit
    vshale = cbw / phit
    vshale = correct_petrophysic_estimation_range(vshale)
    return vshale


_vshale_methods = {
    "linear": vshale_linear,
    "larionov": vshale_larionov,
    "larionov_old": vshale_larionov_old,
    "clavier": vshale_clavier,
    "stieber": vshale_stieber,
    "neu_den": vshale_neu_den,
    "nrm": vshale_nrm
}


def vshale(
    method: Annotated[str, "Chosen vshale method"] = "density", **kwargs) -> np.array:
    """Compute the shale volume from gamma ray log.

    This is a façade for the methods:
        - vshale_linear: :func:`stoneforge.petrophysics.shale_volume.vshale_linear`
        - vshale_larionov: :func:`stoneforge.petrophysics.shale_volume.vshale_larionov`
        - vshale_larionov_old: :func:`stoneforge.petrophysics.shale_volume.vshale_larionov_old`
        - vshale_clavier: :func:`stoneforge.petrophysics.shale_volume.vshale_clavier`
        - vshale_stieber: :func:`stoneforge.petrophysics.shale_volume.vshale_stieber`
        - vshale_neu_den: :func:`stoneforge.petrophysics.shale_volume.vshale_neu_den`
        - vshale_nrm: :func:`stoneforge.petrophysics.shale_volume.vshale_nrm`

    Parameters
    ----------
    gr : array_like
        Gamma Ray log.
    grmin : int, float
        Clean sand GR value.
    grmax : int, float
        Shale/clay value.
    method : str, optional
        Name of the method to be used.  Should be one of the following:
        
            - 'linear'
            - 'larionov'
            - 'larionov_old'
            - 'clavier'
            - 'stieber'
            - 'neu_den'
            - 'nrm'
            
        If not given, default method is 'linear'

    Returns
    -------
    vshale : array_like
        Shale Volume for the aimed interval using the defined method.
    """
    
    options = {}

    required = []
    if method == 'linear':
        required = ['gr', 'grmin', 'grmax']
    elif method == 'larionov':
        required = ['gr', 'grmin', 'grmax']
    elif method == 'larionov_old':
        required = ['gr', 'grmin', 'grmax']
    elif method == 'clavier':
        required = ['gr', 'grmin', 'grmax']
    elif method == 'stieber':
        required = ['gr', 'grmin', 'grmax']
    elif method == 'neu_den':
        required = ['nphi', 'rhob', 'clean_n', 'clean_d', 'fluid_n', 'fluid_d', 'clay_n', 'clay_d']
    elif method == 'nrm':
        required = ['phit', 'phie']
        
    for arg in required:
        if arg not in kwargs:
            msg = f"Missing required argument: {arg}"
            raise ValueError(msg)
        options[arg] = kwargs[arg]
    
    fun = _vshale_methods[method]

    return fun(**options)