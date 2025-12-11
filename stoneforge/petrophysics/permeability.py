# -*- coding: utf-8 -*-

import numpy as np
from typing import Annotated
import warnings
#from stoneforge.petrophysics.helpers import correct_petrophysic_estimation_range

def tixier(
    resd: Annotated[np.array, "Deep resistivity"],
    ress: Annotated[np.array, "Shallow resistivity"], 
    rw: Annotated[np.array, "Water-saturated formation resistivity"]=0.02,
    rhow: Annotated[np.array, "Density of formation water"]=1.10,
    rhoo: Annotated[np.array, "Density of formation oil"]=10000.00,
    inzone: Annotated[np.array, "Invasion radius"]=0.75,
    ) -> np.array:
    """
    Estimate permeability using the empirical method of Tixier (:footcite:t:`tixier1949,mohaghegh1997`).

    Parameters
    ----------
    RESD : array_like
        Deep resistivity in ohm.m.
    RESS : array_like
        Shallow resistivity in ohm.m.
    RW : array_like
        Water-saturated formation resistivity in ohm.m.
    RHOW : array_like
        Density of formation water in g/cm3.
    RHOO : array_like
        Density of formation in g/cm3.
    DDEPTH : array_like
        Invasion radius or change in depth in meters.
    Returns
    -------
    k : array_like
        Estimated permeability (mD) from the Tixier empirical relation.
    """

    dres = resd - ress  # Calculate the resistivity difference

    # Calculate the term inside the parentheses
    term = (2.3 / (rw * (rhow - rhoo))) * (dres / inzone)
    
    # Square the term and multiply by 20
    K = 20 * (term ** 2)
    
    return K