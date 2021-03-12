import numpy as np
import pytest
def phiRHOB(rhob, rhom, rhof, vsh):
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
    phiDtotal : array_like
        Total porosity for the aimed interval using the bulk density.
    phiDeff : array_like
        Effective porosity and shale free for the aimed interval using the bulk density.        
    """

    phiDtotal = (rhom - rhob)/ (rhom - rhof)
    phiDeff = phiDtotal - vsh*0.3
    phiDtotal = np.where(phiDtotal <= 0., 0.,phiDtotal)
    phiDeff = np.where(phiDeff <= 0., 0.,phiDeff)

    return phiDtotal, phiDeff

def phiNPHI(nphi, phi_nsh, vsh):
    """Estimate the effective porosity from the neutron log.

    Parameters
    ----------
    nphi : array_like
        neutron log log.
    phi_nsh : int, float
        Apparent porosity read in the shales on and under the layer under study and with the same values used in φN.
    vsh: int, float
        Total volume of shale in the rock, chosen the most representativev
   
    Returns
    -------

    phin : array_like
        Effective porosity from the neutron log for the aimed interval.
    """

    phin = nphi - (vsh*phi_nsh)
    # phin_cor = phin_cor + (0.04*phin_cor) -> isso seria a correção da matriz

    phin = np.where(phin <= 0., 0.,phin)

    return phin

def phiEff(phiRHOB, phiNPHI):
    """Estimate the effective porosity by calculating the mean of Bulk Density porosity and Neutron porosity.

    Parameters
    ----------
    phiRHOB : array_like
        Effective porosity and shale free for the aimed interval using the bulk density.
    phiNPHI : array_like
        Effective porosity from the neutron log for the aimed interval.
    Returns
    -------

    phie : array_like
        Effective porosity from the Bulk Density porosity and Neutron porosity mean.
    """

    phie = (phiRHOB + phiNPHI)/2
    phie = np.where(phie <= 0., 0.,phie)

    return phie    

def phiSonic(dt, dtma, dtf):
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

    phidt = (dt - dtma)/(dtf - dtma)
    phidt = np.where(phidt <= 0., 0.,phidt)

    return phidt 
