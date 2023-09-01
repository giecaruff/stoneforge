"""
From Berryman 1980
"""
import numpy.typing as npt
import warnings
import numpy as np
from scipy.optimize import fsolve
from stoneforge.rock_physics.elastic_constants import poisson

def get_theta(alpha):
    return alpha*(np.arccos(alpha) - alpha*np.sqrt(1.0 - alpha*alpha))/(1.0 - alpha*alpha)**(3.0/2.0)

def get_f(alpha, theta):
    return (alpha*alpha*(3.0*theta - 2.0))/(1.0 - alpha*alpha)

def PQ(A, B, R, theta, f):
    F1 = 1.0 + A*(1.5*(f + theta) - R*(1.5*f + 2.5*theta - 4.0/3.0))
    F2 = 1.0 + A*(1.0 + 1.5*(f + theta) - R*(1.5*f + 2.5*theta)) + B*(3.0 - 4.0*R) + A*(A + 3.0*B)*(1.5 - 2.0*R)*(f + theta - R*(f - theta + 2.0*theta*theta))
    F3 = 1.0 + A*(1.0 - (f + 1.5*theta) + R*(f + theta))
    F4 = 1.0 + (A/4.0)*(f + 3.0*theta - R*(f - theta))
    F5 = A*(-f + R*(f + theta - 4.0/3.0)) + B*theta*(3.0 - 4.0*R)
    F6 = 1.0 + A*(1.0 + f - R*(f + theta)) + B*(1.0 - theta)*(3.0 - 4.0*R)
    F7 = 2.0 + (A/4.0)*(3.0*f + 9.0*theta - R*(3.0*f + 5.0*theta)) + B*theta*(3.0 - 4.0*R)
    F8 = A*(1.0 - 2.0*R + (f/2.0)*(R - 1.0) + (theta/2.0)*(5.0*R - 3.0)) + B*(1.0 - theta)*(3.0 - 4.0*R)
    F9 = A*((R - 1.0)*f - R*theta) + B*theta*(3.0 - 4.0*R)
    
    P = F1/F2
    Q = (2.0/F3 + 1.0/F4 + ((F4*F5 + F6*F7 - F8*F9)/(F2*F4)))/5.0
    return P, Q

def ABR(k, g, ks, gs):
    vsol = poisson(method='k_and_g', **{"k": ks, "g": gs})
    A = (g/gs) - 1.0
    B = ((k/ks) - (g/gs))/3.0
    R = (1 - (2*vsol))/(2 - (2*vsol))
    return A, B, R



def Kuster_Toksöz(phi: npt.ArrayLike, ks: npt.ArrayLike, gs: npt.ArrayLike, k: float, g: float, alpha: float):
    """
    Calculate bulk modulus and shear modulus using Kuster-Toksöz equation .

    Parameters
    ----------
    phi : array_like
        Porosity log.

    ks : array_like
        Bulk modulus of solid phase.

    gs : array_like
        Shear modulus of solid phase.

    k : float
        Inclusion bulk modulus.

    g : float
        Inclusion shear modulus.

    alpha : float
        Inclusion aspect ratios.

    Returns
    -------
    k_kt : array_like
        Bulk modulus.
    g_kt : array_like
        Shear modulus.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    
    theta = get_theta(alpha)
    f = get_f(alpha, theta)
    A, B, R = ABR(k, g, ks, gs)
    P, Q = PQ(A, B, R, theta, f)

    Fm = (9.0*ks + 8.0*gs)/(ks + 2.0*gs)
    kr = (k - ks)*(phi*P)
    x = (3*kr)/(3*ks + 4*gs)
    numerator = ((4/3)*(gs*x)) + ks
    denominator = (1-x)
    K = numerator / denominator
   
 
  
    #K = ks - (ks + (4.0/3.0)*gs)*phi*(ks - k)*P/3.0/(ks + (4.0/3.0)*gs + phi*(ks - k)*P/3.0)
    G = gs - (gs + Fm)*phi*(gs - g)*Q/5.0/(gs + Fm + phi*(gs - g)*Q/5.0)


   

    return K, G
    

