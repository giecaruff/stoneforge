import warnings
from typing import Callable, Optional, Sequence, Tuple

from scipy import signal
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

def _tcrop(t: npt.ArrayLike) -> npt.ArrayLike:
    """Crop time axis with even number of samples"""
    if len(t) % 2 == 0:
        t = t[:-1]
        warnings.warn("one sample removed from time axis...")
    return t


def Butter_Wavelet(Freq_low: float = 1.5,
                   Freq_hi: float = 65,
                   Samples: int = 71,
                   Dt: float = 4) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    
    r"""Butterworth wavelet

    Create a Butterworth wavelet given time axis ``t``, minimum frequeny and maximun frequency


    Parameters
    ----------
    Freq_low : :obj:`float`, optional
        Minimun frequency
    Freq_hi : :obj:`float`, optional
        Maximun frequency
    Samples : :float
        Number of samples       
    Dt : :float
        Sampling in milisseconds

    Returns
    -------
    wav : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    """

    t = np.arange(Samples)*(Dt/1000)
    t = _tcrop(t)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    imp = signal.unit_impulse(t.shape[0], 'mid')
    b, a = signal.butter(4, Freq_hi,fs = 1000*(1/Dt))
    response_zp = signal.filtfilt(b, a, imp)
    low_b, low_a = signal.butter(2,Freq_low,'hp', fs = 1000*(1/Dt))
    wav = signal.filtfilt(low_b, low_a, response_zp)
    return t, wav

t, butter_wvlt = Butter_Wavelet()
plt.plot(t,butter_wvlt)
plt.xlabel('Tempo (s)')
plt.ylabel('Ondaleta (w)')
plt.grid()
plt.show()
    