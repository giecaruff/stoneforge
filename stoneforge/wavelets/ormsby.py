#Programa que implementa computacionalmente a wavelet de Ormsby

import numpy as np 
import matplotlib.pyplot as plt

def getOrmsby(f,Duration, Sampling):
    assert len(f) == 4, 'Ormsby wavelet needs 4 frequencies as input'
    f = np.sort(f) #Ormsby wavelet frequencies must be in increasing order
    t = np.linspace(-Duration/2,Duration/2,int(Duration/Sampling))
    pif   = np.pi*f
    den1  = pif[3] - pif[2]
    den2  = pif[1] - pif[0]
    term1 = (pif[3]*np.sinc(f[3]*t))**2 - (pif[2]*np.sinc(f[2]*t))**2
    term2 = (pif[1]*np.sinc(f[1]*t))**2 - (pif[0]*np.sinc(f[0]*t))**2

    wav   = term1/den1 - term2/den2;
    wav /= np.amax(wav)
    return t, wav
time, wvlt = getOrmsby((5,10,40,50),0.5,0.004)
plt.plot(time,wvlt)
plt.xlabel('Tempo (s)')
plt.ylabel('Ondaleta (w)')
plt.grid()
plt.show()
    








