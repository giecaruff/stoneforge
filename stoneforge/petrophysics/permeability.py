# -*- coding: utf-8 -*-

import numpy as np
from typing import Annotated
import warnings
#from stoneforge.petrophysics.helpers import correct_petrophysic_estimation_range


def tixier(
    resd: Annotated[np.array, "Deep resistivity"],
    ress: Annotated[np.array, "Shallow resistivity"], 
    rw: Annotated[float, "Water-saturated formation resistivity"]=0.02,
    rhow: Annotated[float, "Density of formation water"]=1.10,
    rhoo: Annotated[float, "Density of formation oil"]=10000.00,
    inzone: Annotated[float, "Invasion radius"]=0.75,
    ) -> np.array:
    """
    Estimate permeability using the empirical method of Tixier (:footcite:t:`tixier1949,mohaghegh1997`).

    Parameters
    ----------
    resd : array_like
        Deep resistivity in ohm.m.
    ress : array_like
        Shallow resistivity in ohm.m.
    rw : float, optional
        Water-saturated formation resistivity in ohm.m.
    rhow : float, optional
        Density of formation water in g/cm3.
    rhoo : float, optional
        Density of formation in g/cm3.
    inzone : array_like, float, optional
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


def timur(
    phi: Annotated[np.array, "Porosity"],
    sw: Annotated[np.array, "Water saturation"]
    ) -> np.array:
    """
    Estimate permeability using the empirical method of Timur (:footcite:t:`timur1968,mohaghegh1997`).

    Parameters
    ----------
    phi : array_like
        Porosity (fraction).
    sw : array_like
        Water saturation (fraction).

    Returns
    -------
    k : array_like
        Estimated permeability (mD) from the Timur empirical relation.
    """

    return (93 * (phi ** 2.2) / (sw)) ** 2


def coates_dumanoir(
    resd: Annotated[np.array, "Deep resistivity"],
    phi: Annotated[np.array, "Porosity"],
    hd: Annotated[float, "Hidrocarbon density"] = 0.8,
    rw: Annotated[float, "Water-saturated formation resistivity"]=0.02,
    ) -> np.ndarray:
    """
    Estimate permeability using the empirical method of Coates and Dumanoir equation for (:footcite:t:`dumoir1973,mohaghegh1997`).

    Parameters
    ----------
    resd : array-like
        Deep (or True) formation resistivity in ohm.m.
    phi : array-like
        Porosity as fraction (m/m).
    hd : array-like
        Hidrocarbon density in g/cm3 (standard for 0.8 as crude oil).
    rw : float, optional
        Formation water resistivity in ohm.m (standard for 0.02).

    Returns
    -------
    k : np.ndarray
        Permeability (in calibration units; e.g. mD if C chosen appropriately).
    """
    # Calculating C constant
    c = 23 + 465 * hd - 188 * hd*hd
    
    # Calculating W constant
    w = np.sqrt((((np.log10(rw / resd) + 2.2) ** 2) / 2.0) + (3.75 - phi))

    # Final permeability
    return ((c * phi ** (2 * w))/((w ** 4) * (rw / resd))) ** 2

def coates(
    phi: Annotated[np.array, "Porosity"],
    sw: Annotated[np.array, "Water saturation"]
    ) -> np.array:
    """
    Estimate permeability using the empirical method of Coates (:footcite:t:`timur1968,schlumberger2013`).

    Parameters
    ----------
    phi : array_like
        Porosity (fraction).
    sw : array_like
        Water saturation (fraction).
        
    Returns
    -------
    k : array_like
        Estimated permeability (mD) from the Timur empirical relation.
    """

    phi = np.asarray(phi, dtype=float)
    swirr = np.asarray(sw, dtype=float)

    # Basic validity checks
    if np.any(phi <= 0) or np.any(phi >= 1):
        print("phi must be in (0, 1) as a fraction.")
    if np.any(swirr <= 0) or np.any(swirr >= 1):
        print("swirr must be in (0, 1) as a fraction.")

    numerator = 100 * phi**2 * (1 - swirr)
    term = numerator / swirr
    K = term**2

    return K