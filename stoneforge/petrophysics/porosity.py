import numpy as np
import pytest


def phi_rhob(rhob, rhom, rhof, vsh):
    """Estimate the porosity from the bulk density.

    Parameters
    ----------
    rhob : array_like
        Bulk density log.
    rhom : int, float
        Matrix density.
    rhof : int, float
        Density of the fluid saturating the rock (Usually 1.0 for water and 1.1 for saltwater mud).
    vsh:array_like
        Shale volume log.
       
    Returns
    -------
    phid_total : array_like
        Total porosity for the aimed interval using the bulk density.
    phid_eff : array_like
        Effective porosity and shale free for the aimed interval using the bulk density.        
    """

    phid_total = (rhom - rhob) / (rhom - rhof)
    phid_eff = phid_total - vsh * 0.3
    phid_total = np.where(phid_total <= 0., 0., phid_total)
    phid_eff = np.where(phid_eff <= 0., 0.,phid_eff)

    return phid_total, phid_eff


def phi_nphi(nphi, nphi_sh, vsh):
    """Estimate the effective porosity from the neutron log.

    Parameters
    ----------
    nphi : array_like
        neutron log.
    phi_nsh : int, float
        Apparent porosity read in the shales on and under the layer under study and with the same values used in φN.
    vsh: int, float
        Total volume of shale in the rock, chosen the most representative.
   
    Returns
    -------

    phin : array_like
        Effective porosity from the neutron log for the aimed interval.
    """

    phin = nphi - (vsh * nphi_sh)
    # phin_cor = phin_cor + (0.04*phin_cor) -> isso seria a correção da matriz

    phin = np.where(phin <= 0., 0., phin)

    return phin


def phi_eff(phi_rhob, phi_nphi):
    """Estimate the effective porosity by calculating the mean of Bulk Density porosity and Neutron porosity.

    Parameters
    ----------
    phi_rhob : array_like
        Effective porosity and shale free for the aimed interval using the bulk density.
    phi_nphi : array_like
        Effective porosity from the neutron log for the aimed interval.
    Returns
    -------

    phie : array_like
        Effective porosity from the Bulk Density porosity and Neutron porosity mean.
    """

    phie = (phi_rhob + phi_nphi) / 2
    phie = np.where(phie <= 0., 0., phie)

    return phie    


def phi_sonic(dt, dtma, dtf):
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
    """

    phidt = (dt - dtma) / (dtf - dtma)
    phidt = np.where(phidt <= 0., 0., phidt)

    return phidt


def phie_gaymard(phid, phin):
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
