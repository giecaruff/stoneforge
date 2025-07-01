# -*- coding: utf-8 -*-

import numpy as np
from typing import Annotated
import warnings
from .helpers import correct_petrophysic_estimation_range

#References
#---------- 
#.. bibliography::

def effective_porosity(
    phi: Annotated[np.array, "Porosity log"],
    vsh: Annotated[np.array, "Shale volume"]) -> np.array:
    """Calculate the effective porosity from the total porosity and shale volume (:footcite:t:`schon1998physical`).

    Parameters
    ----------
    phi : array_like
        Porosity log
    vsh : array_like
        Shale volume

    Returns
    -------
    phie : array_like
        Effective porosity for the aimed interval (more suitable for the bulk density porosity)
    """
    
    phie = phi - vsh
    phie = correct_petrophysic_estimation_range(phie)
    return phie


def density_porosity(
    rhob: Annotated[np.array, "Bulk density log"],
    rhom: Annotated[float, "Matrix density"],
    rhof: Annotated[float, "Fluid density"]) -> np.array:
    
    """Estimate the porosity from the bulk density log (:footcite:t:`schon1998physical`).

    Parameters
    ----------
    rhob : array_like
        Bulk density log.
    rhom : float
        Matrix density.
    rhof : float
        Density of the fluid saturating the rock (Usually 1.0 for water and 1.1 for saltwater mud).
       
    Returns
    -------
    phid : array_like
        Total porosity based on bulk density.

    """
    if rhom == rhof:
        warnings.warn(UserWarning("This will result in a division by zero"))

        return np.nan

    elif rhom < rhof or any(rhom <= rhob):
        warnings.warn(UserWarning("rhom must be greater than rhof and rhob"))

        phi = (rhom - rhob) / (rhom - rhof)

    elif any(rhom - rhob > rhom - rhof):
        warnings.warn(UserWarning("rhob value is lower than rhof"))

        phi = (rhom - rhob) / (rhom - rhof)

    else: 
        phi = (rhom - rhob) / (rhom - rhof)


    phi = correct_petrophysic_estimation_range(phi)
    return phi


def neutron_porosity(
    nphi: Annotated[np.array, "Neutron porosity log"],
    vsh: Annotated[np.array, "Shale volume"],
    phish: Annotated[float, "Apparent porosity in shales"]) -> np.array:
    """Estimate the effective porosity from the neutron log (:footcite:t:`schon1998physical`).

    Parameters
    ----------
    nphi : array_like
        Neutron porosity log.
    vsh : array_like
        Total volume of shale in the rock, chosen the most representative.
    phi_nsh : int, float
        Apparent porosity read in the shales on and under the layer under study and with the same values used in φN.

    Returns
    -------
    phin : array_like
        Effective porosity from the neutron log for the aimed interval.

    """
    if any(nphi < (vsh * phish)):
        warnings.warn(UserWarning("phin must be a positive value"))
        phin = nphi - (vsh * phish)
    
    elif any(nphi - (vsh * phish) > 1):
        warnings.warn(UserWarning("phin must be a value between 0 and 1"))
        phin = nphi - (vsh * phish)

    else:
        phin = nphi - (vsh * phish)

    phin = correct_petrophysic_estimation_range(phin)
    return phin


def neutron_density_porosity(
    phid: Annotated[np.array, "Porosity from density log"],
    phin: Annotated[np.array, "Porosity from neutron log"],
    squared: Annotated[bool, "Main operation"]=False) -> np.array:
    """Estimate the effective porosity by calculating the mean of Bulk Density porosity and Neutron porosity (:footcite:t:`schon1998physical`).

    Parameters
    ----------
    phid : array_like
        Effective porosity and shale free for the aimed interval using the bulk density.
    phin : array_like
        Effective porosity from the neutron log for the aimed interval.
    squared : bool, optional
        If True, the porosity is calculated using the square root of the mean of the squares of the two porosities.
        If False, the porosity is calculated using the mean of the two porosities. Default is False.

    Returns
    -------
    phie : array_like
        Effective porosity from the Bulk Density porosity and Neutron porosity mean.

    """
    if squared == False:
        if any((phid + phin / 2) > 1):
            warnings.warn(UserWarning("phi must be a value between 0 and 1"))
            phi = (phid + phin) / 2
        else:
            phi = (phid + phin) / 2
    elif squared == True:
        if any((phid**2 + phin**2 / 2) > 1):
            warnings.warn(UserWarning("phi must be a value between 0 and 1"))
            phi = np.sqrt( (phid**2 + phin**2) / 2)
        else:
            phi = np.sqrt( (phid**2 + phin**2) / 2)
    phi = correct_petrophysic_estimation_range(phi)
    return phi


def sonic_porosity(
    dt: Annotated[np.array, "Sonic log"],
    dtma: Annotated[np.array, "Matrix transit time"],
    dtf: Annotated[np.array, "Fluid transit time"]) -> np.array:
    
    """Estimate the Porosity from sonic using the :footcite:t:`wyllie1956` time-average equation.

    Parameters
    ----------
    dt : array_like
        Sonic log reading (acoustic transit time (μsec/ft))
    dtma : int, float
        Acoustic transit time of the matrix (μsec/ft)
    dtf : int, float
        Acoustic transit time of the fluids, usually water (μsec/ft)
              
    Returns
    -------
    phidt : array_like
        Porosity from sonic.

    """
    if dtf == dtma:
        warnings.warn(UserWarning("This will result in a division by zero"))
        return np.nan

    elif any(dt <= dtma) or dtf <= dtma:
        warnings.warn(UserWarning("dt and dtf must be greater than dtma"))
        phidt = (dt - dtma) / (dtf - dtma)

    elif any(dt - dtma > dtf - dtma):
        warnings.warn(UserWarning("dt value is greather than dtf"))
        phidt = (dt - dtma) / (dtf - dtma)

    else:
        phidt = (dt - dtma) / (dtf - dtma)
        
    phidt = correct_petrophysic_estimation_range(phidt)
    return phidt


def gaymard_porosity(
    phid: Annotated[np.array, "Porosity from density log"],
    phin: Annotated[np.array, "Porosity from neutron log"]) -> np.array:
    
    """Estimate the effective porosity using :footcite:t:`gaymardpoupon1968` method.

    Parameters
    ----------
    phid : array_like
        Density porosity (porosity calculated using density log)
    phin : int, float
        Neutron porosity (porosity calculated using neutron log)

    Returns
    -------
    phie : array_like
        Effective porosity using Gaymard-Poupon method

    """
    phie = (0.5 * (phid*phid + phin*phin)) ** 0.5

    phie = correct_petrophysic_estimation_range(phie)
    return phie


_porosity_methods = {
    "density": density_porosity,
    "neutron": neutron_porosity,
    "neutron-density": neutron_density_porosity,
    "sonic": sonic_porosity,
    "gaymard": gaymard_porosity,
    "effective": effective_porosity
}

def porosity(
    method: Annotated[str, "Chosen porosity method"] = "density", **kwargs) -> np.array:
    """Compute porosity from well logs.

    This is a façade for the methods:
        - density: :func:`stoneforge.petrophysics.porosity.density_porosity`
        - neutron: :func:`stoneforge.petrophysics.porosity.neutron_porosity`
        - neutron-density: :func:`stoneforge.petrophysics.porosity.neutron_density_porosity`
        - sonic: :func:`stoneforge.petrophysics.porosity.sonic_porosity`
        - gaymard: :func:`stoneforge.petrophysics.porosity.gaymard_porosity`
        - effective: :func:`stoneforge.petrophysics.porosity.effective_porosity`

    Parameters
    ----------
    rhob : array_like
        Bulk density log. Required if `method` is "denisty".
    rhom : int, float
        Matrix density. Required if `method` is "denisty".
    rhof : int, float
        Density of the fluid saturating the rock (Usually 1.0 for water and 1.1 for saltwater mud). Required if `method` is "denisty".
    nphi : array_like
        Neutron log. Required if `method` is "neutron".
    vsh : array_like
        Total volume of shale in the rock, chosen the most representative. Required if `method` is "neutron" or "effective".
    phi_nsh : int, float
        Apparent porosity read in the shales on and under the layer under study and with the same values used in φN. Required if `method` is "neutron".
    dt : array_like
        Sonic log reading (acoustic transit time (μsec/ft)). Required if `method` is "sonic".
    dtma : int, float
        Acoustic transit time of the matrix (μsec/ft). Required if `method` is "sonic".
    dtf : int, float
        Acoustic transit time of the fluids, usually water (μsec/ft). Required if `method` is "sonic".
    phid : array_like
        Density porosity (porosity calculated using density log). Required if `method` is "neutron-density" or "gaymard.
    phin : int, float
        Neutron porosity (porosity calculated using neutron log). Required if `method` is "neutron-density" or "gaymard.
    phi : int, float
        Total porisity. Required if `method` is "effective".
    method : str, optional
        Name of the method to be used.  Should be one of
        
            - 'density'
            - 'neutron'
            - 'neutron-density'
            - 'sonic'
            - 'gaymard'
            - 'effective'
            
        If not given, default method is 'density'

    Returns
    -------
    phi : array_like
        Porosity log using the defined method.
        
    """
    options = {}

    required = []
    if method == "density":
        required = ["rhob", "rhom", "rhof"]
    elif method == "neutron":
        required = ["nphi", "vsh", "phish"]
    elif method == "neutron-density":
        required = ["phid", "phin"]
    elif method == "sonic":
        required = ["dt", "dtma", "dtf"]
    elif method == "gaymard":
        required = ["phid", "phin"]
    elif method == "effective":
        required = ["phi", "vsh"]

    for arg in required:
        if arg not in kwargs:
            msg = f"Missing required argument for method '{method}': '{arg}'"
            raise TypeError(msg)
        options[arg] = kwargs[arg]

    fun = _porosity_methods[method]

    return fun(**options)
