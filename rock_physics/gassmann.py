import logging
import pytest


def Kdry(porosity, Ks, KsatA, KfluidA):
    """Calculate the dry-rock bulk modulus using Gassmann's equation (1951).

    Parameters
    ----------
    porosity : int, float, array_like
        Porosity value or log with values between 0 and 1.

    Ks : int, float, array_like
        Bulk modulus of solid phase in Pascal.

    KsatA : int, float, array_like
        Bulk modulus of the rock saturated with fluid A in Pascal.

    KfluidA : int, float, array_like
        Bulk modulus of the fluid A in Pascal.

    Returns
    -------
    Kdry : int, float, array_like
        Dry-rock bulk modulus in Pascal.
    """

    if (porosity < 0.0) or (porosity > 1.0):
        msg = "There are invalid values of porosity log."
        logging.warning(msg)

    if (Ks < 10**5) or (KsatA < 10**5) or (KfluidA < 10**5):
        # Just to warn. If True, the unity maybe is in GPa.
        msg = "The unity of Ks, KsatA and KfluidA should be in Pascal."
        logging.warning(msg)

    phi = porosity
    Kdry_num = 1 - (1 - phi) * (KsatA/Ks) - (phi * KsatA/KfluidA)
    Kdry_den = 1 + phi - (phi*Ks / KfluidA) - (KsatA/Ks)
    Kdry = Ks * (Kdry_num / Kdry_den)

    return Kdry
    

def Ksat(porosity, Ks, Kdry, KfluidB):
    """Calculate the bulk modulus of the rock saturated with fluid B.

    Parameters
    ----------
    porosity : int, float, array_like
        Porosity value or log with values between 0 and 1.

    Ks : int, float, array_like
        Bulk modulus of solid phase in Pascal.

    Kdry : int, float, array_like
        Bulk modulus of the dry-rock in Pascal.

    KfluidB : int, float, array_like
        Bulk modulus of the fluid B in Pascal.

    Returns
    -------
    Ksat : int, float, array_like
        Bulk modulus of the rock saturated with fluid B in Pascal.
    """
    
    if (porosity < 0.0) or (porosity > 1.0):
        msg = "There are invalid values of porosity log."
        logging.warning(msg)

    if (Ks < 10**5) or (Kdry < 10**5) or (KfluidB < 10**5):
        # Just to warn. If True, the unity maybe is in GPa.
        msg = "The unity of Ks, Kdry and KfluidB should be in Pascal."
        logging.warning(msg)

    phi = porosity
    Ksat_num = phi*Kdry - (1 + phi)*(KfluidB * Kdry / Ks) + KfluidB
    Ksat_den = (1 - phi)*KfluidB + phi*Ks - (KfluidB * Kdry / Ks)
    Ksat = Ks * (Ksat_num / Ksat_den)

    return Ksat

