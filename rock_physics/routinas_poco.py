# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 08:17:05 2018

@author: lupin
"""
import numpy as np
import math

def velocity(DT):
    
    """
    Funcao para tranformar DT (us/f) para velocidade (m/s)
    """
    
    Vp = 304800/DT
    
    return Vp

def gardner(Vp, a, b):
    
    """
    Funcao para a densidade a partir da velocidade usando o modelo de Gardner
    
    """
    
    RHOB_gard = a*Vp**b
    
    return RHOB_gard

def densidade_Vp_polinomial(Vp, m):
    
    """
    Funcao para a densidade a partir da velocidade usando o modelo de Brocher (2005)
    """
    
   
    nl = max(np.shape(Vp))
    
    G = np.ones((nl,1))
    
    for pw in range(1,len(m)):
    
        G = np.c_[G, Vp**pw ]
        
        
        
    RHOB = np.dot(G,m)
    
    return RHOB

"""
Funcoes para estimar o volume de argila usando diferentes metodos
"""

def volclay(GR, GRmin, GRmax):
    """
    Funcao para estimar a volume de argila a partir do modelo linear
    """
#    if math.isnan(GR) : GR=0        #Se tiver um nan assume GR=0
    Vcl = (GR-GRmin)/(GRmax-GRmin)
    
    Vcl[Vcl>1.0] = 1.0
    Vcl[Vcl<0.0] = 0.0
    return Vcl*100

def volclay_larionov(GR, GRmin, GRmax):
    """
    Funcao para estimar a volume de argila a partir do modelo de Larionov
    Referencia: Larionov, W. W., 1969, Borehole radiometry: Nedra.
    """
    
    IGR = (GR-GRmin)/(GRmax-GRmin)
    IGR[IGR>1.0] = 1.0
    IGR[IGR<0.0] = 0.0
    a = 0.0083
    b = 3.7
    Vcl_larionov = a*(2**(b*IGR)-1)*100
    
    return Vcl_larionov
    
    
    
    
    

