import numpy as np

def soft_sand(K, G, phi, phic, n, P):
    
    """
    Funcao para calcular o modelo de Soft Sand
    Entrada: K e G modulos de bulk e cisalhante do mineral
    phi porosidade, phic porosidade critica e n numero de coordenacao (numero de contatos),
    P pressão hidrostática de confinamento
    Ksoft e Gsoft curvas dos modulos de bulk e cisalhante para o modelo Soft Sand
    """
    
    v = (3.*K-2.*G)/(2.*(3.*K+G))
    Khm = ((n**2. * (1.-phic)**2. * G**2 * P)/( 18.*np.pi**2 * (1.-v)**2))**(1./3.)
    Ghm = ((5.-4.*v)/(5.*(2.-v))) * ((3.*n**2 * (1.-phic)**2. * G**2. * P)/( 2*np.pi**2. * (1.-v)**2))**(1./3.)
    
    zhm = (Ghm/6.) * (9.*Khm + 8.*Ghm)/(Khm + 2.*Ghm)
    
    Ksoft = ((phi/phic)/(Khm + 4./3.* Ghm) + (1. - phi/phic)/(K + 4./3.*Ghm))**-1. - 4./3. * Ghm
    Gsoft = ((phi/phic)/(Ghm + zhm) + (1. - phi/phic)/(G + zhm))**-1. - zhm
    
    return Ksoft, Gsoft

def const_soft_sand(K, G, Kb, Gb, phi, phib, n, P):
    
    """
    Funcao para calcular o modelo de Constant Cement Sand
    Entrada: K e G modulos de bulk e cisalhante do mineral
    phi porosidade, phic porosidade critica e n numero de coordenacao (numero de contatos),
    P pressão hidrostática de confinamento
    Konst e Gsoft curvas dos modulos de bulk e cisalhante para o modelo Constant Cement Sand
    """
    zb = (Gb/6.) * (9.*Kb + 8.*Gb)/(Kb + 2.*Gb)
    Kconst_soft = ((phi/phib)/(Kb + 4./3.* Gb) + (1. - phi/phib)/(K + 4./3.*Gb))**-1. - 4./3. * Gb
    Gconst_soft = ((phi/phib)/(Gb + zb) + (1. - phi/phib)/(G + zb))**-1. - zb
    
    return Kconst_soft, Gconst_soft

def const_sand(K, G, Kcem, Gcem, phi, phic, n, P):
    
    """
    Funcao para calcular o modelo de Constant Cement Sand
    Entrada: K e G modulos de bulk e cisalhante do mineral
    phi porosidade, phic porosidade critica e n numero de coordenacao (numero de contatos),
    P pressão hidrostática de confinamento
    Konst e Gsoft curvas dos modulos de bulk e cisalhante para o modelo Constant Cement Sand
    """
    zcem = (Gcem/6.) * (9.*Kcem + 8.*Gcem)/(Kcem + 2.*Gcem)
    Kconst = ((phi/phic)/(Kcem + 4./3.* Gcem) + (1. - phi/phic)/(K + 4./3.*Gcem))**-1. - 4./3. * Gcem
    Gconst = ((phi/phic)/(Gcem + zcem) + (1. - phi/phic)/(G + zcem))**-1. - zcem
    
    return Kconst, Gconst

def stiff_sand(K, G, phi, phic, n, P):
    
    """
    Funcao para calcular o modelo de Stiff Sand
    Entrada: K e G modulos de bulk e cisalhante do mineral
    phi porosidade, phic porosidade critica e n numero de coordenacao (numero de contatos),
    P pressão hidrostática de confinamento
    Ksoft e Gsoft curvas dos modulos de bulk e cisalhante para o modelo Soft Sand
    """
    
    v = (3.*K-2.*G)/(2.*(3.*K+G))
    Khm = ((n**2. * (1.-phic)**2. * G**2 * P)/( 18.*np.pi**2 * (1.-v)**2))**(1./3.)
    Ghm = ((5.-4.*v)/(5.*(2.-v))) * ((3.*n**2 * (1.-phic)**2. * G**2. * P)/( 2*np.pi**2. * (1.-v)**2))**(1./3.)
    
    z = (G/6.) * (9.*K + 8.*G)/(K + 2.*G)
    
    Kstiff = ((phi/phic)/(Khm + 4./3.* G) + (1. - phi/phic)/(K + 4./3.*G))**-1. - 4./3. * G
    Gstiff = ((phi/phic)/(Ghm + z) + (1. - phi/phic)/(G + z))**-1. - z
    
    return Kstiff, Gstiff

def cement_sand(K, G, Kc, Gc, phi, phic, n):
    
    """
    Funcao para calcular o modelo de Contact-Cement Sand
    Entrada: K e G modulos de bulk e cisalhante do mineral,  Kc e Gc modulos de bulk e cisalhante do cimento
    phi porosidade, phic porosidade critica e n numero de coordenacao (numero de contatos)
    Kcem e Gcem curvas do modelo Contact-Cement Sand
    """
    
    v = 0.5 * (K/G -2./3.) / (K/G + 1./3.) 
    vc = 0.5 * (Kc/Gc -2./3.) / (Kc/Gc + 1./3.)
    
    Lbn = (2. * Gc / (np.pi * G)) * (((1. - v) * (1. - vc)) / (1. - 2. * vc))
    Lbt = Gc/(np.pi * G)
    alpha = ((2. * (phic - phi)) / (3. * (1 - phic)))**0.5
    
    At = (-10.**-2) * (2.26 * v**2 + 2.07 * v + 2.3) * Lbt**(0.079 * v**2 + 0.1754 * v - 1.342)
    Bt = (0.0573 * v**2 + 0.0937 * v + 0.202) * Lbt**(0.0274 * v**2 + 0.0529 * v - 0.8765)
    Ct = 10**-4*(9.654 * v**2 + 4.945 * v + 3.1) * Lbt**(0.01867 * v**2 + 0.4011 * v - 1.8186)
    
    St = At * alpha**2 + Bt * alpha + Ct
    
    An = (-0.024153) * Lbn **-1.3646
    Bn = (0.20405) * Lbn**-0.89008
    Cn = (0.00024649) * Lbn**-1.9864
    
    Sn = (An * alpha**2) + (Bn * alpha) + Cn
    
    Mc = Kc + 4./3. * Gc
    
    Kcem = 1./6. * n * (1.-phic) * Mc * Sn
    Gcem = (3./5. * Kcem ) + (3./20. * n * (1.-phic) * Gc * St)
        
    
    return Kcem, Gcem