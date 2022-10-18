def young(method: str, **kwargs):
    """Computes the Young' modulus (e) using two or more elastic moduli [1]_.

    Parameters
    ----------
    method : str
        The two elastic moduli used to the estimation. Should be one of:
            - "v_and_k" for Poisson' ratio and bulk modulus
            - "v_and_g" for Poisson' ratio and shear modulus
            - "v_and_lamda" for Poisson' ratio and Lamé parameter
            - "k_and_g" for bulk and shear moduli
            - "k_and_lamda" for bulk modulus and Lamé parameter
            - "g_and_lamda" for shear modulus and Lamé parameter
            - "vp_and_vs_and_rho" for P-wave velocity, S-wave velocity, and density

    Returns
    -------
    e : int, float, array_like
        Young' modulus.
    
    References
    ----------
    .. [1] Simm, R., & Bacon, M. (2014). Seismic Amplitude: An Interpreter's Handbook.
    Cambridge University Press.

    """
    if method == "v_and_k":
        required = ["v", "k"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return 3*kwargs["k"] * (1 - 2*kwargs["v"])

    if method == "v_and_g":
        required = ["v", "g"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return 2*kwargs["g"] * (1 + kwargs["v"])

    if method == "v_and_lamda":
        required = ["v", "lamda"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return (kwargs["lamda"] / kwargs["v"]) * (1 + kwargs["v"]) * (1 - 2*kwargs["v"])

    if method == "k_and_g":
        required = ["k", "g"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return (9 * kwargs["k"] * kwargs["g"]) / (3 * kwargs["k"] + kwargs["g"])

    if method == "k_and_lamda":
        required = ["k", "lamda"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return 9*kwargs["k"] * (kwargs["k"] - kwargs["lamda"]) / (3*kwargs["k"] - kwargs["lamda"])

    if method == "g_and_lamda":
        required = ["g", "lamda"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return kwargs["g"] * (3*kwargs["lamda"] + 2*kwargs["g"]) / (kwargs["lamda"] + kwargs["g"])

    if method == "vp_and_vs_and_rho":
        required = ["vp", "vs", "rho"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return kwargs["rho"] * kwargs["vs"]**2 * (3 * kwargs["vp"]**2 - 4 * kwargs["vs"]**2) / (kwargs["vp"]**2 - kwargs["vs"]**2)
        


def poisson(method: str, **kwargs):
    """Computes the Poisson' ratio (v) using two or more elastic moduli [1]_.

    Parameters
    ----------
    method : str
        The two elastic moduli used to the estimation. Should be one of:
            - "e_and_k" for Young' and bulk moduli
            - "e_and_g" for Young' and shear moduli
            - "k_and_g" for bulk and shear moduli
            - "k_and_lamda" for bulk modulus and Lamé parameter
            - "g_and_lamda" for shear modulus and Lamé parameter
            - "vp_and_vs" for P-wave and S-wave velocities
            

    In the case where "k_and_g" is selected, the kwargs must contain both
    k and g parameters (kwargs = ["k":36 * 10**9, "g":45 * 10**9])

    Returns
    -------
    v : int, float, array_like
        Poisson' ratio.

    References
    ----------
    .. [1] Simm, R., & Bacon, M. (2014). Seismic Amplitude: An Interpreter's Handbook.
    Cambridge University Press.

    """
    if method == "e_and_k":
        required = ["e", "k"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return (3*kwargs["k"] - kwargs["e"]) / (6*kwargs["k"])
    
    if method == "e_and_g":
        required = ["e", "k"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return (kwargs["e"] - 2*kwargs["g"]) / 2*kwargs["g"]
    
    if method == "k_and_g":
        required = ["k", "g"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return (3*kwargs["k"] - 2*kwargs["g"]) / (6*kwargs["k"] + 2*kwargs["g"])

    if method == "k_and_lamda":
        required = ["k", "lamda"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return kwargs["lamda"] / (3*kwargs["k"] - kwargs["lamda"])
    
    if method == "g_and_lamda":
        required = ["g", "lamda"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return kwargs["lamda"] / 2*(kwargs["lamda"] + kwargs["g"])

    if method == "vp_and_vs":
        required = ["vp", "vs"]
        for arg in required:
            if arg not in kwargs:
                msg = f"Missing required argument for method '{method}': '{arg}'"
                raise TypeError(msg)
        return (kwargs["vp"]**2 - 2*(kwargs["vs"]**2)) / 2*(kwargs["vp"]**2 - kwargs["vs"]**2) 
