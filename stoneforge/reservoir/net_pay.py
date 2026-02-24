# -*- coding: utf-8 -*-

import numpy as np
from typing import Annotated
import warnings
from ..petrophysics.helpers import correct_petrophysic_estimation_range

def net_pay_siliciclastic(
        vsh: Annotated[np.array, "shaliness log data"],
        phi: Annotated[np.array, "porosity log data"],
        sw: Annotated[np.array, "water saturation log data"],
        vsh_t: Annotated[np.array, "shaliness threshold"] = 0.3,
        phi_t: Annotated[np.array, "porosity threshold"] = 0.20,
        sw_t: Annotated[np.array, "water saturation threshold"] = 0.3,
        fillzeros: Annotated[bool, "if fill values with zeros; default is False"] = False):
    
    """Calculates the net pay for siliciclastic reservoirs based on shale volume, porosity, and water saturation logs (:footcite:t:`crain1999,girao2013`).
    
    Parameters
    ----------
    vsh : array_like
        Shaliness log data (dimensionless).
    phi : array_like
        Porosity log data (dimensionless).
    sw : array_like
        Water saturation log data (dimensionless).
    vsh_t : float, optional
        Shaliness threshold value. Default is 0.3.
    phi_t : float, optional
        Porosity threshold value. Default is 0.20.
    sw_t : float, optional
        Water saturation threshold value. Default is 0.3.
    fillzeros : bool, optional
        If True, values that do not meet the cutoff conditions will be filled with zeros. If

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'rock': An array indicating the rock presence based on the shaliness threshold.
        - 'res': An array indicating the reservoir presence based on the porosity threshold and rock presence.
        - 'pay': An array indicating the pay zone based on the water saturation threshold and reservoir presence.
    """
    
    rock = cutoff(vsh, t=vsh_t, fillzeros=fillzeros)
    res = cutoff(phi, t=phi_t, lower=False, fillzeros=fillzeros)*rock
    pay = cutoff(sw, t=sw_t, fillzeros=fillzeros)*res

    return {'rock': rock, 'res': res, 'pay': pay}

def cutoff(
        log: Annotated[np.array, "well log data"],
        t: Annotated[float, "threshold value"],
        lower: Annotated[bool, "whether to apply threshold on lower values; default is True"]=True,
        fillzeros: Annotated[bool, "if fill values with zeros; default is False"]=False,
        equal: Annotated[bool, "whether to include equal values in the threshold; default is False"]=False)-> np.array:
    
    """Applies a cutoff to a well log data based on a specified threshold value (:footcite:t:`crain1999,girao2013`). 
    
    Parameters
    ----------
    log : array_like
        The well log data to which the cutoff will be applied.
    t : float
        The threshold value for the cutoff.
    lower : bool, optional
        Whether to apply the cutoff on lower values (True) or higher values (False). Default is True.
    fillzeros : bool, optional
        If True, values that do not meet the cutoff condition will be filled with zeros. If False, they will be filled with NaN. Default is False.
    equal : bool, optional
        Whether to include values equal to the threshold in the cutoff condition. Default is False.

    Returns
    -------
    array_like
        An array where values that meet the cutoff condition are set to 1.0, and values that do not meet the condition are set to 0.0 or NaN based on the fillzeros parameter.

    Notes
    -----
    The function applies a cutoff to the input log data based on the specified threshold value `t`. The cutoff is applied either to values below or above the threshold, depending on the `lower` parameter. If `equal` is True, values equal to the threshold are included in the cutoff condition. If `fillzeros` is True, values that do not meet the cutoff condition are filled with zeros; otherwise, they are filled with NaN.
    """

    log = np.asarray(log, dtype=float)

    if lower:
        mask = log <= t if equal else log < t
    else:
        mask = log >= t if equal else log > t

    if fillzeros:
        return np.where(mask, 1.0, 0.0)

    return np.where(mask, 1.0, np.nan)