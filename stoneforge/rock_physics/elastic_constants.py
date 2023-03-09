import numpy.typing as npt

def bulk_modulus(rhob: npt.ArrayLike, vp: npt.ArrayLike, vs: npt.ArrayLike):
     """
     Computes the bulk modulus using density, shear wave velocity and compressional wave velocity.

    Parameters
    rhob : array_like, float
        Density data in g/cm³.

    vp : array_like, float
        Compressional wave velocity data in km/s.

    vs : array_like, float
        Shear wave velocity data in km/s.

    Returns
    -------
    K : array_like, float
        Bulk modulus data in Pascal unit.
    ----------

    References
    ----------
    .. [1] Simm, R., & Bacon, M. (2014). Seismic Amplitude: An Interpreter's Handbook.
    Cambridge University Press.

    """
     A = vp**2 
     B = (4/3) * (vs**2)
     K = rhob * (A - B)
     return K

def shear_modulus(rhob: npt.ArrayLike, vp: npt.ArrayLike):
    """
     Computes the shear modulus using density and s wave velocity.

    Parameters
    rhob : array_like, float
        Density data in g/cm³.

    vp : array_like, float
        Compressional wave velocity data in km/s.

    Returns
    -------
    U : array_like, float
        Shear modulus data in Pascal unit.
    ----------

    References
    ----------
    .. [1] Simm, R., & Bacon, M. (2014). Seismic Amplitude: An Interpreter's Handbook.
    Cambridge University Press.

    """
    U = rhob * (vp**2)

    return U

def shear_modulus(rhob: npt.ArrayLike, vp: npt.ArrayLike):
    """
     Computes the shear modulus using density and s wave velocity.

    Parameters
    rhob : array_like, float
        Density data in g/cm³.

    vp : array_like, float
        Compressional wave velocity data in km/s.

    Returns
    -------
    U : array_like, float
        Shear modulus data in Pascal unit.
    ----------

    References
    ----------
    .. [1] Simm, R., & Bacon, M. (2014). Seismic Amplitude: An Interpreter's Handbook.
    Cambridge University Press.

    """
    U = rhob * (vp**2)

    return U

def shear_wave_velocity(rhob: npt.ArrayLike, u: npt.ArrayLike):
    """
     Computes the shear wave velocity using density and shear modulus.

    Parameters
    rhob : array_like, float
        Density data in g/cm³.

    u: array_like, float
        Shear modulus data in Pascal unit.

    Returns
    -------
    VS : array_like, float
        Shear wave velocity data km/s.
    ----------

    References
    ----------
    .. [1] Simm, R., & Bacon, M. (2014). Seismic Amplitude: An Interpreter's Handbook.
    Cambridge University Press.

    """
    VS = (rhob / u)**0.5

    return VS


def compressional_wave_velocity(method: str, **kwargs):
    """
     Computes the compressional wave velocity.

    Parameters
    method : str
        The two elastic moduli used to the estimation. Should be one of:
            - "rhob_and_g_and_k" for density and shear moduli and bolk moduli
            - "rhob_and_m" for density and compressional modulus
    ----------
    Returns
    -------
    VP : int, float, array_like
        compressional wave velocity.

    References
    ----------
    .. [1] Simm, R., & Bacon, M. (2014). Seismic Amplitude: An Interpreter's Handbook.
    Cambridge University Press.
  

    """
    if method == "rhob_and_g_and_k":
        required = ["rhob", "g", "k"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        A = kwargs["k"] + (4/3)*kwargs["g"]
        VP = (A / kwargs["rhob"])**0.5
        return VP
    
    elif method == "rhob_and_m":
        required = ["rhob", "m"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        VP = (kwargs["m"]/kwargs["rhob"])
        return VP



def compressional_modulus(rhob: npt.ArrayLike, vp: npt.ArrayLike):
    """
     Computes the Compressional modulus using density and compressional wave velocity

    Parameters
    rhob : array_like, float
        Density data in g/cm³.

    vp: array_like, float
        Compressional wave velocity in km/s².


    Returns
    -------
    M : array_like, float
        Compressional modulus data kg/m³.
    ----------

    References
    ----------
    .. [1] Simm, R., & Bacon, M. (2014). Seismic Amplitude: An Interpreter's Handbook.
    Cambridge University Press.

    """
    M = rhob * (vp)**2

    return M

                          


def poisson(method: str, **kwargs):
    """Computes the Poisson ratio using two elastic moduli.

    Parameters
    ----------
    method : str
        The two elastic moduli used to the estimation. Should be one of:
            - "k_and_g" for bulk and shear moduli
            - "vp_and_vs" for P-wave and S-wave velocities
            - "e_and_k" for Young's and bulk moduli

    In the case where "k_and_g" is selected, the kwargs must contain both
    k and g parameters (kwargs = ["k":36, "g":45])

    Returns
    -------
    v : int, float, array_like
        Poisson ratio.

    References
    ----------
    .. [1] Simm, R., & Bacon, M. (2014). Seismic Amplitude: An Interpreter's Handbook.
    Cambridge University Press.

    """
    if method == "k_and_g":
        required = ["k", "g"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return (3*kwargs["k"] - 2*kwargs["g"]) / (6*kwargs["k"] + 2*kwargs["g"])

    if method == "vp_and_vs":
        required = ["vp", "vs"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return (kwargs["vp"]**2 - 2*(kwargs["vs"]**2)) / 2*(kwargs["vp"]**2 - kwargs["vs"]**2) 

    if method == "e_and_k":
        required = ["e", "k"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return (3*kwargs["k"] - kwargs["e"]) / (6*kwargs["k"])