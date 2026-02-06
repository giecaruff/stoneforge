# -*- coding: utf-8 -*-

import numpy as np
from typing import Annotated
import warnings
#from stoneforge.petrophysics.helpers import correct_petrophysic_estimation_range
from .helpers import correct_petrophysic_estimation_range

    
def archie(
    rt: Annotated[np.array, "Formation resistivity"],
    phi: Annotated[np.array, "Porosity"],
    rw: Annotated[float, "Water resistivity"] = 0.02,
    a: Annotated[float, "Clean density point"] = 1.00,
    m: Annotated[float, "fluid neutron point"] = 2.00,
    n: Annotated[float, "fluid density point"] = 2.00) -> np.array:
    """Estimate the Water Saturation from :footcite:t:`archie1942` (standard values from :footcite:t:`archie1952,aapg2014archie`).

    Parameters
    ----------
    rt : array_like
        Formation resistivity.    
    phi : array_like
        Porosity.
    rw : float
        Water resistivity.  
    a : float
        Tortuosity factor.
    m : float
        Cementation exponent.
    n : float
        Saturation exponent.

    Returns
    -------
    sw : array_like
        Water saturation from Archie equation.

    """
    if np.any(((a*rw) / (phi**m * rt))**(1/n) > 1):
        warnings.warn(UserWarning("saturation of water must be a value between 0 and 1"))
        sw = ((a*rw) / (phi**m * rt))**(1/n)
        sw = correct_petrophysic_estimation_range(sw)
        return sw

    else:
        sw = ((a*rw) / (phi**m * rt))**(1/n)
        sw = correct_petrophysic_estimation_range(sw)
        return sw


def simandoux(
    rt: Annotated[np.array, "Formation resistivity"],
    phi: Annotated[np.array, "Porosity"],
    vsh: Annotated[np.array, "Shale volume"],
    rw: Annotated[float, "Water resistivity"] = 0.02,
    rsh: Annotated[float, "Shale resistivity"] = 4.00,
    a: Annotated[float, "Clean density point"] = 1.00,
    m: Annotated[float, "fluid neutron point"] = 2.00,
    n: Annotated[float, "fluid density point"] = 2.00) -> np.array:
    """Estimate water saturation from :footcite:t:`simandoux1963` equation (standard values from :footcite:t:`geoloil2012sw,aapg2014archie`).

    Parameters
    ---------- 
    phi : array_like
        Porosity.
    vsh : array_like
        Clay volume log.
    rw : float
        Water resistivity.
    rsh : float
        Clay resistivity.
    rt : array_like
        True resistivity.   
    a : float
        Tortuosity factor.
    m : float
        Cementation exponent.
    n : float
        Saturation exponent.

    Returns
    -------
    sw : array_like
        Water saturation from Simandoux equation.

    """
    C = (1 - vsh) * a * rw / phi**m
    D = C * vsh / (2*rsh)
    E = C / rt
    sw = ((D**2 + E)**0.5 - D)**(2/n)

    sw = correct_petrophysic_estimation_range(sw)

    return sw


def indonesia(
    rt: Annotated[np.array, "Formation resistivity"],
    phi: Annotated[np.array, "Porosity"],
    vsh: Annotated[np.array, "Shale volume"],
    rw: Annotated[float, "Water resistivity"] = 0.02,
    rsh: Annotated[float, "Shale resistivity"] = 4.00,
    a: Annotated[float, "Clean density point"] = 1.00,
    m: Annotated[float, "fluid neutron point"] = 2.00,
    n: Annotated[float, "fluid density point"] = 2.00) -> np.array:
    """Estimate water saturation from :footcite:t:`poupon-leveaux1971` equation (standard values from :footcite:t:`geoloil2012sw,aapg2014archie`).

    Parameters
    ---------- 
    phi : array_like
        Porosity.
    vsh : array_like
        Clay volume log.
    rw : float
        Water resistivity.
    rsh : float
        Clay resistivity.
    rt : array_like
        True resistivity.   
    a : float
        Tortuosity factor.
    m : float
        Cementation exponent.
    n : float
        Saturation exponent.

    Returns
    -------
    indonesia : array_like
        Water saturation from Poupon-Leveaux equation.

    """
    #sw = ((1/rt)**0.5 / ((vsh**(1 - 0.5*vsh) / (rsh)**0.5) + (phi**m / a*rw)**0.5))**(2/n)
    C = (1./rt) ** 0.5
    D = 1 - 0.5*vsh
    E = (vsh**D)/(rsh**0.5)
    F = ((phi**m)/(a*rw))**0.5
    sw = (C/(E+F))**(1/n)
    sw = correct_petrophysic_estimation_range(sw)

    return sw


def fertl(
    rt: Annotated[np.array, "Formation resistivity"],
    phi: Annotated[np.array, "Porosity"],
    vsh: Annotated[np.array, "Shale volume"],
    rw: Annotated[float, "Water resistivity"] = 0.02,
    a: Annotated[float, "Clean density point"] = 1.00,
    m: Annotated[float, "fluid neutron point"] = 2.00,
    alpha: Annotated[float, "fluid density point"] = 0.30) -> np.array:
    """Estimate water saturation from :footcite:t:`fertl1975` equation (standard values from :footcite:t:`aapg2014archie`).

    Parameters
    ----------
    rt : array_like
        True resistivity.
    phi : array_like
        Porosity (must be effective).
    vsh : array_like
        Clay volume log.
    rw : float
        Water resistivity.     
    a : int, float
        Tortuosity factor.
    m : int, float
        Cementation exponent.
    alpha : float
        Alpha parameter from Fertl equation.

    Returns
    -------
    fertl : array_like
        Water saturation from Fertl equation.
        
    """
    sw = phi**(-m/2) * ((a*rw/rt + (alpha*vsh/2)**2)**0.5 - (alpha*vsh/2))
    sw = correct_petrophysic_estimation_range(sw)


    return sw


_sw_methods = {
    "archie": archie,
    "simandoux": simandoux,
    "indonesia": indonesia,
    "fertl": fertl
}


def water_saturation(rw: float, rt: np.array, phi: np.array,
                     a: float, m: float, method: str = "archie",
                     **kwargs) -> np.array:
    """Compute water saturation from resistivity log.

    This is a façade for the methods:
        - archie
        - simandoux
        - indonesia
        - fertl

    Parameters
    ----------
    rw : int, float
        Water resistivity.
    rt : array_like
        True resistivity.
    phi : array_like
        Porosity (must be effective).
    a : int, float
        Tortuosity factor.
    m : int, float
        Cementation exponent.
    n : int, float
        Saturation exponent. Required if `method` is "archie", "simandoux" or
        "indonesia".
    vsh : array_like
        Clay volume log. Required if `method` is "simandoux", "indonesia" or
        "fertl".
    rsh : float
        Clay resistivity. Required if `method` is "simandoux" or "indonesia".
    alpha : array_like
        Alpha parameter from Fertl equation. Required if `method` is "fertl"
    method : str, optional
        Name of the method to be used.  Should be one of
        
            - 'archie'
            - 'simandoux'
            - 'indonesia'
            - 'fertl'
            
        If not given, default method is 'archie'

    Returns
    -------
    water_saturation : array_like
        Water saturation for the aimed interval using the defined method.

    """
    options = {}
    
    required = []
    if method == "archie":
        required = ["n"]
    elif method == "simandoux":
        required = ["n", "vsh", "rsh"]
    elif method == "indonesia":
        required = ["n", "vsh", "rsh"]
    elif method == "fertl":
        required = ["vsh", "alpha"]
    
    for arg in required:
        if arg not in kwargs:
            msg = f"Missing required argument for method '{method}': '{arg}'"
            raise TypeError(msg)
        options[arg] = kwargs[arg]
    
    fun = _sw_methods[method]


    sw = fun(rw, rt, phi, a, m, **options)
    
    return sw
