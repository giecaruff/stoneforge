# -*- coding: utf-8 -*-

import numpy as np
from typing import Annotated

def bulk_modulus(
    rho: Annotated[np.array, "Density data"],
    vp: Annotated[np.array, "Compressional velocity data"],
    vs: Annotated[np.array, "Shear velocity data"])-> np.array:
    """Computes the bulk modulus using density, shear wave velocity and compressional wave velocity (:footcite:t:`simm2014`).

    Parameters
    ----------
    rho : array_like
        Density data in g/cm³.

    vp : array_like
        Compressional wave velocity data in km/s.

    vs : array_like
        Shear wave velocity data in km/s.

    Returns
    -------
    K : array_like
        Bulk modulus data in Pascal unit (S/I).

    """
    A = vp**2 
    B = (4/3) * (vs**2)
    return rho * (A - B)


def compressional_modulus(
    rho: Annotated[np.array, "Density data"],
    vp: Annotated[np.array, "Compressional velocity data"])-> np.array:
    """Computes the Compressional modulus using density and compressional wave velocity (:footcite:t:`simm2014`).

    Parameters
    ----------
    rho : array_like, float
        Density data in g/cm³.

    vp: array_like, float
        Compressional wave velocity in km/s².

    Returns
    -------
    M : array_like, float
        Compressional modulus data Pascal (Pa).
        
    """
    return rho * (vp)**2


def shear_modulus(
    rho: Annotated[np.array, "Density data"],
    vs: Annotated[np.array, "Shear velocity data"]) -> np.array:
    """Computes the shear modulus using density and s wave velocity (:footcite:t:`simm2014`).

    Parameters
    ----------
    rho : array_like
        Density data in g/cm³.

    vs : array_like
        Compressional wave velocity data in km/s.

    Returns
    -------
    U : array_like
        Shear modulus data in Pascal unit.
        
    """
    return rho * (vs**2)


def shear_wave_velocity(
    rho: Annotated[np.array, "Density data"],
    u: Annotated[np.array, "Shear modulus data"]) -> np.array:
    """Computes the shear wave velocity using density and shear modulus (:footcite:t:`simm2014`).

    Parameters
    ----------
    rho : array_like, float
        Density data in g/cm³.

    u: array_like, float
        Shear modulus data in Pascal unit.

    Returns
    -------
    vs : array_like, float
        Shear wave velocity data km/s.
        
    """
    return (u / rho)**0.5


def compressional_wave_velocity(
    method: Annotated[str, "Chosen method to compute compressional wave velocity"] = "rhob_and_g_and_k", **kwargs) -> np.array:
    """
    Compute the compressional (P-wave) velocity using different elastic property methods (:footcite:t:`simm2014`).

    Parameters
    ----------
    method : {'rhob_and_g_and_k', 'rhob_and_m'}
        Method used to compute compressional wave velocity.
        - 'rhob_and_g_and_k' : Requires bulk modulus (K), shear modulus (G), and density (RHOB).
        - 'rhob_and_m'       : Requires compressional modulus (M) and density (RHOB).
    
    rhob : float or array_like
        Bulk density of the medium [g/cm³ or kg/m³ depending on units consistency].
    
    g : float or array_like, (optional)
        Shear modulus (G) [same unit system as K]. Required for 'rhob_and_g_and_k'.
    
    k : float or array_like, (optional)
        Bulk modulus (K) [Pa or similar]. Required for 'rhob_and_g_and_k'.
    
    m : float or array_like, (optional)
        Compressional modulus (M = K + 4G/3) [Pa]. Required for 'rhob_and_m'.

    Returns
    -------
    vp : float or array_like
        Compressional wave velocity [m/s] or consistent unit.

    Raises
    ------
    TypeError
        If required parameters are missing for the selected method.

    Examples
    --------
    >>> compressional_wave_velocity('rhob_and_g_and_k', rhob=2.5, g=30e9, k=45e9)
    6000.0

    >>> compressional_wave_velocity('rhob_and_m', rhob=2.5, m=75e9)
    6000.0
    """
    def from_rhob_g_k(rhob, g, k):
        return ((k + (4 / 3) * g) / rhob) ** 0.5

    def from_rhob_m(rhob, m):
        return (m / rhob) ** 0.5

    method_map = {
        "rhob_and_g_and_k": (from_rhob_g_k, ["rhob", "g", "k"]),
        "rhob_and_m": (from_rhob_m, ["rhob", "m"]),
    }

    if method not in method_map:
        raise ValueError(f"Unsupported method '{method}'. Choose from: {list(method_map.keys())}")

    func, required_args = method_map[method]
    missing = [arg for arg in required_args if arg not in kwargs]
    if missing:
        raise TypeError(f"Missing required arguments for method '{method}': {', '.join(missing)}")

    return func(**{key: kwargs[key] for key in required_args})


def poisson(
    method: Annotated[str, "Chosen method to compute Poisson's ratio"] = "k_and_g", **kwargs) -> np.array:
    """
    Compute Poisson's ratio (ν) using different pairs of elastic moduli (:footcite:t:`simm2014`).

    Parameters
    ----------
    method : {'k_and_g', 'vp_and_vs', 'e_and_k'}
        Method used to compute Poisson's ratio:
        - 'k_and_g'    : Requires bulk modulus (K) and shear modulus (G).
        - 'vp_and_vs'  : Requires compressional (P-wave) velocity (VP) and shear (S-wave) velocity (VS).
        - 'e_and_k'    : Requires Young's modulus (E) and bulk modulus (K).
    
    k : float or array_like, (optional)
        Bulk modulus [Pa or similar]. Required for 'k_and_g' and 'e_and_k'.
    
    g : float or array_like, (optional)
        Shear modulus [Pa]. Required for 'k_and_g'.

    e : float or array_like, (optional)
        Young's modulus [Pa]. Required for 'e_and_k'.
    
    vp : float or array_like, (optional)
        Compressional wave velocity [m/s]. Required for 'vp_and_vs'.
    
    vs : float or array_like, (optional)
        Shear wave velocity [m/s]. Required for 'vp_and_vs'.

    Returns
    -------
    v : float or array_like
        Poisson's ratio (dimensionless).

    Raises
    ------
    TypeError
        If required parameters for the selected method are missing.

    Examples
    --------
    >>> poisson('k_and_g', k=36e9, g=45e9)
    0.2

    >>> poisson('vp_and_vs', vp=6000, vs=3464)
    0.25

    >>> poisson('e_and_k', e=90e9, k=36e9)
    0.2
    """
    def from_k_and_g(k, g):
        return (3 * k - 2 * g) / (6 * k + 2 * g)

    def from_vp_and_vs(vp, vs):
        return ((vp**2 - 2 * vs**2) / (2 * (vp**2 - vs**2)))

    def from_e_and_k(e, k):
        return (3 * k - e) / (6 * k)

    method_map = {
        "k_and_g": (from_k_and_g, ["k", "g"]),
        "vp_and_vs": (from_vp_and_vs, ["vp", "vs"]),
        "e_and_k": (from_e_and_k, ["e", "k"]),
    }

    if method not in method_map:
        raise ValueError(f"Unsupported method '{method}'. Choose from: {list(method_map.keys())}")

    func, required_args = method_map[method]
    missing = [arg for arg in required_args if arg not in kwargs]
    if missing:
        raise TypeError(f"Missing required arguments for method '{method}': {', '.join(missing)}")

    return func(**{key: kwargs[key] for key in required_args})