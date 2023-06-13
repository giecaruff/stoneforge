#Programa que implementa computacionalmente a wavelet de Ricker

import numpy as np 
import matplotlib.pyplot as plt

def Ricker_Wavelet(Peak_freq, Duration, Sampling) :
    t = np.linspace(-Duration/2,Duration/2,int(Duration/Sampling))
    ricker = (1. -2.*(np.pi**2)*(Peak_freq**2)*(t**2))*np.exp(-(np.pi**2)*(Peak_freq**2)*(t**2))
    return t, ricker
Time , Ricker_wl = Ricker_Wavelet(20,0.5,0.004)

plt.plot(Time, Ricker_wl)
plt.xlabel('Tempo (s)')
plt.ylabel('Ondaleta (w)')
plt.grid()
plt.show()

