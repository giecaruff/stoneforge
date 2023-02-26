def bulk_modulus(rhob, vp, vs):
     """
     Computes the bulk modulus using density, shear wave velocity and compressional wave velocity.

    Parameters
    rhob : array_like, float
        Density data.

    vp : array_like, float
        Compressional wave velocity data.

    vs : array_like, float
        Shear wave velocity data.

    Returns
    -------
    K : array_like, float
        Bulk modulus data.
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