
import numpy as np

def calculate_toc_from_passey(dt, logrt, dtbaseline, logrtbaseline, lom=10.6):

    """Estimate the Total Organic Carbon Content by Passey method using Sonic log and Resistivy log _.

    Parameters
    ----------
    dt : array_like
        Sonic log reading (acoustic transit time (μsec/ft))
    logrt : array_like
        Resistivity log reading (formation resistivity (ohm/m))
    dtbaseline : int, float
        Sonic log base line (μsec/ft)
    logrtbaseline : int, float
        Resistivity log base line (ohm/m)
    lom : int, float
        Level of maturity
              
    Returns
    -------
    TOC : array_like
        Total organic carbon content calculated from passey method.

    References
    ----------
    Passey, O.R., F.U. Moretti, and J.D. Stroud, 1990, A practical modal for organic richness from porosity and resistivity logs: AAPG Bulletin, v. 
    74, p. 1777–1794.


    """
    dlogrt = (logrt - logrtbaseline) + 0.02*(dt - dtbaseline)
    toc = dlogrt*10**(2.297 - 0.1688*lom)
    return np.clip(toc, 0.0, 100.0)