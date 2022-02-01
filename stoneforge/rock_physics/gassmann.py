import numpy as np
import numpy.typing as npt
from typing import Union


def Kdry(phi: Union[float, npt.ArrayLike], Ks: Union[float, npt.ArrayLike],
         KsatA: Union[float, npt.ArrayLike],
         KfluidA: Union[float, npt.ArrayLike]) -> np.ndarray:
    """Calculate the dry-rock bulk modulus using Gassmann's [1]_ equation .

    Parameters
    ----------
    phi : float, array_like
        Porosity value or log.

    Ks : float, array_like
        Bulk modulus of solid phase.

    KsatA : float, array_like
        Bulk modulus of the rock saturated with fluid.

    KfluidA : float, array_like
        Bulk modulus of the fluid A.

    Returns
    -------
    Kdry : float, array_like
        Dry-rock bulk modulus.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    Kdry_num = 1 - (1 - phi) * (KsatA/Ks) - (phi * KsatA/KfluidA)
    Kdry_den = 1 + phi - (phi*Ks / KfluidA) - (KsatA/Ks)
    Kdry = Ks * (Kdry_num / Kdry_den)

    return Kdry
    

def Ksat(phi: Union[float, npt.ArrayLike], Ks: Union[float, npt.ArrayLike],
         Kdry: Union[float, npt.ArrayLike],
         KfluidB: Union[float, npt.ArrayLike]) -> np.ndarray:
    """Calculate the bulk modulus of the rock saturated with fluid B [1]_.

    Parameters
    ----------
    phi : float, array_like
        Porosity value or log with values between 0 and 1.

    Ks : float, array_like
        Bulk modulus of solid phase in Pascal.

    Kdry : float, array_like
        Bulk modulus of the dry-rock in Pascal.

    KfluidB : float, array_like
        Bulk modulus of the fluid B in Pascal.

    Returns
    -------
    Ksat : float, array_like
        Bulk modulus of the rock saturated with fluid B.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    Ksat_num = phi*Kdry - (1 + phi)*(KfluidB * Kdry / Ks) + KfluidB
    Ksat_den = (1 - phi)*KfluidB + phi*Ks - (KfluidB * Kdry / Ks)
    Ksat = Ks * (Ksat_num / Ksat_den)

    return Ksat
