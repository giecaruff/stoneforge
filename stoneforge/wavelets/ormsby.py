
import warnings
from typing import Sequence, Tuple

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt


def _tcrop(t: npt.ArrayLike) -> npt.ArrayLike:
    """Crop time axis with even number of samples"""
    if len(t) % 2 == 0:
        t = t[:-1]
        warnings.warn("one sample removed from time axis...")
    return t

def getOrmsby(f: Sequence[float] = (5,10,40,50),
              Samples: float = 71, 
              Dt: float = 4) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    r"""Ormsby wavelet
    Create a Ormsby wavelet given time axis ``t`` and frequency range
    defined by four frequencies which parametrize a trapezoidal shape in
    the frequency spectrum.
    Parameters
    ----------
    f : :obj:`tuple`, optional
        Frequency range
    Samples : :obj:`float`, optional
        Number of samples       
    Dt : obj:`float`, optional
        Sampling in milisseconds
    Returns
    -------
    wav : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    """
    assert len(f) == 4, 'Ormsby wavelet needs 4 frequencies as input'
    f = np.sort(f) #Ormsby wavelet frequencies must be in increasing order
    t = np.arange(Samples)*(Dt/1000)
    t = _tcrop(t)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    pif   = np.pi*f
    den1  = pif[3] - pif[2]
    den2  = pif[1] - pif[0]
    term1 = (pif[3]*np.sinc(f[3]*t))**2 - (pif[2]*np.sinc(f[2]*t))**2
    term2 = (pif[1]*np.sinc(f[1]*t))**2 - (pif[0]*np.sinc(f[0]*t))**2

    wav   = term1/den1 - term2/den2;
    wav /= np.amax(wav)
    return t, wav


