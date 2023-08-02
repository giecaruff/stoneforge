#Programa que implementa computacionalmente a wavelet de Ricker

import warnings
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt


def _tcrop(t: npt.ArrayLike) -> npt.ArrayLike:
    """Crop time axis with even number of samples"""
    if len(t) % 2 == 0:
        t = t[:-1]
        warnings.warn("one sample removed from time axis...")
    return t

def Ricker_Wavelet(Peak_freq: float = 30,
                   Samples: float = 71,
                   Dt: float = 4) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    r"""Ricker wavelet

    Create a Ricker wavelet given time axis ``t`` and central frequency ``f_0``

    Parameters
    ----------
    Peak_freq : :obj:`float`, optional
        Central frequency
    Samples : :obj:`float`, optional
        Number of samples
    Dt : :obj:`func`, optional
        Sampling in milisseconds

    Returns
    -------
    ricker : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    
    """
    t = np.arange(Samples)*(Dt/1000)
    t = _tcrop(t)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    ricker = (1. -2.*(np.pi**2)*(Peak_freq**2)*(t**2))*np.exp(-(np.pi**2)*(Peak_freq**2)*(t**2))
    return t, ricker
Time , Ricker_wl = Ricker_Wavelet()

plt.plot(Time, Ricker_wl)
plt.xlabel('Tempo (s)')
plt.ylabel('Ondaleta (w)')
plt.grid()
plt.show()

