import numpy as np
import pytest
def VShale_Linear(gr, grmin, grmax, depth, top, bottom):
    """Estimate the shale volume from the linear model.
    Parameters
    ----------
    gr : array_like
        Gamma Ray log.
    grmin : int, float
        Clean sand GR value.
    grmax : int, float
        Shale/clay value.
    depth : array_like
        Depth log corresponding to the GR data.        
    top : int, float
        Top of range to be calculated.
    bottom : int, float
        Bottom of range to be calculated.        
    Returns
    -------
    VShale_Lin : array_like
        Shale Volume for the aimed interval using the Linear method.
    """ 

    interval = (depth > top) & (depth < bottom)
    VShale_Lin = (gr[interval] - grmin) / (grmax - grmin)

    return VShale_Lin

def VShale_Larionov(gr, grmin, grmax, depth, top, bottom, tertiary=True):
    """Estimate the shale volume from the Larionov model for Tertiary or older rocks.
    Parameters
    ----------
    gr : array_like
        Gamma Ray log.
    grmin : int, float
        Clean sand GR value.
    grmax : int, float
        Shale/clay value.
    depth : array_like
        Depth log corresponding to the GR data.        
    top : int, float
        Top of range to be calculated.
    bottom : int, float
        Bottom of range to be calculated.  
    tertiary: boolean 
        Choose whether it will be the account for Tertiary or older rocks  .            
    Returns
    -------
    VShale_Lari : array_like
        Shale Volume for the aimed interval using the Larionov method.
    """
    interval = (depth > top) & (depth < bottom)

    if tertiary == True:
        igr = (gr[interval] - grmin) / (grmax - grmin)
        VShale_Lari = 0.083 * (2 ** (3.7 * igr) - 1)
    else:
        igr = (gr[interval] - grmin) / (grmax - grmin)
        VShale_Lari = 0.33 * (2. ** (2. * igr) - 1)

    return VShale_Lari