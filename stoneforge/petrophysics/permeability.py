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

    return 0.136 * ((phi**4.4) / (sw**2))


def coates_dumanoir(
    resd: Annotated[np.array, "Deep resistivity"],
    phi: Annotated[np.array, "Porosity"],
    C: Annotated[float, "Empirical calibration constant"] = 1.0,
    w: Annotated[float, "Empirical porosity resistivity constant"] = None,
    rw: Annotated[float, "Water-saturated formation resistivity"]=0.02,
    compute_w_as_sqrt: bool = True
    ) -> np.ndarray:
    """
    Estimate permeability using the empirical method of Coates and Dumanoir equation for (:footcite:t:`dumoir1973,mohaghegh1997`).

    Parameters
    ----------
    phi : array-like
        Porosity as fraction (0..1).
    swirr : array-like
        Irreducible water saturation (fraction, 0..1).
    C : float, optional
        Empirical calibration constant (no units by itself; k units depend on C),
        default 1.0 (user should set according to calibration).
    w : array-like or float, optional
        The 'w' parameter used in the original formula. If provided, this is used
        directly. If omitted, and both rw and resd are provided, `w` is computed
        from the Coates-Dumanoir relation for w^2 (see below).
    rw : float, optional
        Formation water resistivity. Required only if `w` is not provided and you
        want to compute `w` from resistivities.
    resd : array-like or float, optional
        Resistivity at irreducible water saturation (resd). Required if computing w.
    compute_w_as_sqrt : bool, optional
        When computing w from w^2, set True to take w = sqrt(w^2) (default).
        If you prefer to use w^2 directly in the formula, set False (less common).

    Returns
    -------
    k : np.ndarray
        Permeability (in calibration units; e.g. mD if C chosen appropriately).
    """

    phi = np.asarray(phi, dtype=float)

    # Determine w:
    if w is None:
        if (rw is None) or (resd is None):
            raise ValueError("Either 'w' or both 'rw' and 'resd' must be provided.")
        resd = np.asarray(resd, dtype=float)
        # compute w^2 from Coates & Dumanoir formula:
        w2 = 3.75 - phi + 0.5 * (np.log10(rw / resd) + 2.2)**2
        if compute_w_as_sqrt:
            # take principal square root (w >= 0)
            w = np.sqrt(np.maximum(w2, 0.0))
        else:
            # use w^2 directly - user must understand effect on formula
            w = w2
    else:
        w = np.asarray(w, dtype=float)

    # Prevent zeros in denominator w^4
    if np.any(w == 0):
        raise ValueError("Computed or supplied 'w' contains zero values (would divide by zero).")

    # compute sqrt(k)
    sqrt_k = (C / (w**4)) * ( ((phi) ** (2 * w))/ (rw/resd))

    # final permeability
    k = sqrt_k**2 
    return np.asarray(k, dtype=float)

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