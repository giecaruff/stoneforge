import numpy as np
import numpy.typing as npt


def density_porosity(rhob: npt.ArrayLike, rhom: float, rhof: float) -> np.ndarray:
    """Estimate the porosity from the bulk density log [1]_.

    Parameters
    ----------
    rhob : array_like
        Bulk density log.
    rhom : int, float
        Matrix density.
    rhof : int, float
        Density of the fluid saturating the rock (Usually 1.0 for water and 1.1 for saltwater mud).
       
    Returns
    -------
    phi : array_like
        Total porosity for the aimed interval using the bulk density.

    References
    ----------      
    .. [1] Schön, J. H. (2015). Physical properties of rocks: Fundamentals and 
    principles of petrophysics. Elsevier.

    """
    phi = (rhom - rhob) / (rhom - rhof)

    return phi


def neutron_porosity(nphi: npt.ArrayLike, vsh: npt.ArrayLike,
             nphi_sh: float) -> np.ndarray:
    """Estimate the effective porosity from the neutron log [1]_.

    Parameters
    ----------
    nphi : array_like
        neutron log.
    vsh : array_like
        Total volume of shale in the rock, chosen the most representative.
    phi_nsh : int, float
        Apparent porosity read in the shales on and under the layer under study and with the same values used in φN.

    Returns
    -------
    phin : array_like
        Effective porosity from the neutron log for the aimed interval.

    References
    ----------
    .. [1] Schön, J. H. (2015). Physical properties of rocks: Fundamentals and 
    principles of petrophysics. Elsevier.

    """
    phin = nphi - (vsh * nphi_sh)

    return phin


def neutron_density_porosity(phid: npt.ArrayLike, phin: npt.ArrayLike,
                squared: bool = False) -> np.ndarray:
    """Estimate the effective porosity by calculating the mean of Bulk Density porosity and Neutron porosity [1]_.

    Parameters
    ----------
    phid : array_like
        Effective porosity and shale free for the aimed interval using the bulk density.
    phin : array_like
        Effective porosity from the neutron log for the aimed interval.

    Returns
    -------
    phie : array_like
        Effective porosity from the Bulk Density porosity and Neutron porosity mean.

    References
    ----------
    TODO

    """
    if squared == False:
        phi = (phid + phin) / 2
    elif squared == True:
        phi = np.sqrt( (phid**2 + phin**2) / 2)

    return phi  


#TODO phit -> phie (clay volume correction)


def sonic_porosity(dt, dtma, dtf):
    """Estimate the Porosity from sonic using the Wyllie time-average equatio (http://dx.doi.org/10.1190/1.1438217).

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

    References
    ----------
    TODO

    """
    phidt = (dt - dtma) / (dtf - dtma)

    return phidt


def gaymard_porosity(phid, phin):
    """Estimate the effective porosity using Gaymard-Poupon [1]_ method.

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
    
    References
    ----------
    .. [1] Gaymard, R., and A. Poupon. "Response Of Neutron And Formation
    Density Logs In Hydrocarbon Bearing Formations." The Log Analyst 9 (1968).

    """
    phie = (0.5 * (phid*phid + phin*phin)) ** 0.5

    return phie
