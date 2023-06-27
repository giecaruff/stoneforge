import numpy as np
import numpy.typing as npt

from stoneforge.rock_physics.elastic_constants import poisson


def hertz_mindlin(k: float, g: float, n: float, phic: float,
                  p: float) -> np.ndarray:
    """Computes the elastic moduli of the original room-dry grain pack at 
    critical porosity phic from Hertz-Mindlin contact theory [1]_.

    Parameters
    ----------
    k : int, float
        Bulk modulus of the mineral.
    g : int, float
        Shear modulus of the mineral.
    n : int, float
        Coordination number.
    phic : float
        Critical porosity.
    p : int, float
        Hydrostatic confining pressure.

    Returns
    -------
    khm : float
        Bulk modulus of the Hertz-Mindlin point.
    ghm : float
        Shear modulus of the Hertz-Mindlin point.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    v = poisson(method = "k_and_g", k=k, g=g)
    khm = ((n**2 * (1.-phic)**2 * g**2 * p) / \
          ( 18*np.pi**2 * (1-v)**2))**(1/3)
    ghm = ((5-4*v)/(5*(2-v))) * ((3*n**2 * (1-phic)**2 * g**2 * p) / \
          ( 2*np.pi**2 * (1-v)**2))**(1/3)

    return khm, ghm


def soft_sand(k: float, g: float, phi: npt.ArrayLike, phic: float,
              n: float, p: float) -> np.ndarray:
    """Computes the elastic moduli of the rock using the soft sand model [1]_.

    Parameters
    ----------
    k : int, float
        Bulk modulus of the mineral.
    g : int, float
        Shear modulus of the mineral.
    phi : float, array_like
        Porosity value or log.
    phic : float
        Critical porosity.
    n : int, float
        Coordination number.
    p : int, float
        Hydrostatic confining pressure.

    Returns
    -------
    ksoft : float, array_like
        Bulk modulus using the soft sand model.
    gsoft : float, array_like
        Shear modulus using the soft sand model.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    khm, ghm = hertz_mindlin(k, g, n, phic, p)
    zhm = (ghm/6) * (9*khm + 8*ghm)/(khm + 2*ghm)
    ksoft = ((phi/phic)/(khm + 4/3* ghm) + (1 - phi/phic) / \
            (k + 4/3*ghm))**-1 - 4/3 * ghm
    gsoft = ((phi/phic)/(ghm + zhm) + (1 - phi/phic)/(g + zhm))**-1 - zhm
    
    return ksoft, gsoft


def constant_cement(k: float, g: float, phi: npt.ArrayLike, phic: float,
              n: float, kc: float, gc: float, phib: float) -> np.ndarray:
    """Computes the elastic moduli of the rock using the constant cement
    model [1]_.

    Parameters
    ----------
    k : int, float
        Bulk modulus of the mineral.
    g : int, float
        Shear modulus of the mineral.
    phi : float, array_like
        Porosity value or log.
    phic : float
        Critical porosity.
    n : int, float
        Coordination number.
    kc : int, float
        Bulk modulus of the cementing mineral.
    gc : int, float
        Shear modulus of the cementing mineral.
    phib : float
        Porosity where the cement effect starts.

    Returns
    -------
    kconst : float, array_like
        Bulk modulus using the constant cement model.
    gconst : float, array_like
        Shear modulus using the constant cement model.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    if isinstance(phi, float):
        kconst, gconst = np.zeros((1)), np.zeros((1))
    else:
        kconst, gconst = np.zeros(phi.shape), np.zeros(phi.shape)
    
    soft_domain = phi < phib
    cement_domain = phi >= phib

    kcem, gcem = contact_cement(k, g, phi, phic, n, kc, gc)
    kend, gend = contact_cement(k, g, phib, phic, n, kc, gc)
    
    if not isinstance(phi, float):
        kconst[cement_domain], gconst[cement_domain] = kcem[cement_domain], gcem[cement_domain]
        
        kb, gb = kend, gend
        zb = (gb/6) * (9*kb + 8*gb)/(kb + 2*gb)
        kco = ((phi/phib)/(kb + 4/3* gb) + (1 - phi/phib)/(k + 4/3*gb))**-1 - 4/3 * gb
        gco = ((phi/phib)/(gb + zb) + (1 - phi/phib)/(g + zb))**-1 - zb
        kconst[soft_domain], gconst[soft_domain] = kco[soft_domain], gco[soft_domain]    
    
    else:
        if cement_domain:
            kconst, gconst = kcem, gcem
        else:
            kb, gb = kend, gend
            zb = (gb/6) * (9*kb + 8*gb)/(kb + 2*gb)
            kco = ((phi/phib)/(kb + 4/3* gb) + (1 - phi/phib)/(k + 4/3*gb))**-1 - 4/3 * gb
            gco = ((phi/phib)/(gb + zb) + (1 - phi/phib)/(g + zb))**-1 - zb
            kconst, gconst = kco, gco

    return kconst, gconst


def stiff_sand(k: float, g: float, phi: npt.ArrayLike, phic: float,
               n: float, p: float) -> np.ndarray:
    """Computes the elastic moduli of the rock using the stiff sand model [1]_.

    Parameters
    ----------
    k : int, float
        Bulk modulus of the mineral.
    g : int, float
        Shear modulus of the mineral.
    phi : float, array_like
        Porosity value or log.
    phic : float
        Critical porosity.
    n : int, float
        Coordination number.
    p : int, float
        Hydrostatic confining pressure.

    Returns
    -------
    kstiff : float, array_like
        Bulk modulus using the stiff sand model.
    gstiff : float, array_like
        Shear modulus using the stiff sand model.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    khm, ghm = hertz_mindlin(k, g, n, phic, p)
    z = (g/6) * (9*k + 8*g)/(k + 2*g)
    kstiff = ((phi/phic)/(khm + 4/3* g) + (1 - phi/phic)/(k + 4/3*g))**-1 - 4/3 * g
    gstiff = ((phi/phic)/(ghm + z) + (1 - phi/phic)/(g + z))**-1 - z
    
    return kstiff, gstiff


def contact_cement(k: float, g: float, phi: npt.ArrayLike, phic: float,
                   n: float, kc: float, gc: float, deposition_type: str = 'grain_surface') -> np.ndarray:
    """Computes the elastic moduli of the rock using the contact cement
    model [1]_.

    Parameters
    ----------
    k : int, float
        Bulk modulus of the mineral.
    g : int, float
        Shear modulus of the mineral.
    phi : float, array_like
        Porosity value or log.
    phic : float
        Critical porosity.
    n : int, float
        Coordination number.
    kc : int, float
        Bulk modulus of the cementing mineral.
    gc : int, float
        Shear modulus of the cementing mineral.
    gc : deposition_type
        Cement deposition framework: grain_surface or grain_contact

    Returns
    -------
    kcem : float, array_like
        Bulk modulus using the contact cement model.
    gcem : float, array_like
        Shear modulus using the contact cement model.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    v = poisson(method = "k_and_g", k=k, g=g)
    vc = poisson(method = "k_and_g", k=kc, g=gc)
    
    Lbn = (2 * gc / (np.pi * g)) * (((1 - v) * (1 - vc)) / (1 - 2 * vc))
    Lbt = gc/(np.pi * g)
    if deposition_type == 'grain_surface':
        alpha = ((2 * (phic - phi)) / (3 * (1 - phic)))**0.5
    elif deposition_type == 'grain_contact':
        alpha = 2*(((phic - phi) / (3 * n * (1 - phic)))**0.25)
    
    At = (-10**-2) * (2.26 * v**2 + 2.07 * v + 2.3) * Lbt**(0.079 * v**2 + 0.1754 * v - 1.342)
    Bt = (0.0573 * v**2 + 0.0937 * v + 0.202) * Lbt**(0.0274 * v**2 + 0.0529 * v - 0.8765)
    Ct = 10**-4*(9.654 * v**2 + 4.945 * v + 3.1) * Lbt**(0.01867 * v**2 + 0.4011 * v - 1.8186)
    
    St = At * alpha**2 + Bt * alpha + Ct
    
    An = (-0.024153) * Lbn **-1.3646
    Bn = (0.20405) * Lbn**-0.89008
    Cn = (0.00024649) * Lbn**-1.9864
    
    Sn = (An * alpha**2) + (Bn * alpha) + Cn
    
    Mc = kc + 4/3 * gc
    
    kcem = 1/6 * n * (1.-phic) * Mc * Sn
    gcem = (3/5 * kcem ) + (3/20 * n * (1-phic) * gc * St)
        
    return kcem, gcem


_gem_models = {
    "soft_sand": soft_sand,
    "stiff_sand": stiff_sand,
    "contact_cement": contact_cement,
    "constant_cement": constant_cement
}


def gem(k: float, g: float, phi: npt.ArrayLike, phic: float, n: float,
        method: str = "soft_sand", **kwargs) -> np.ndarray:
    """Computes one of the granular effective medium rock-physics models.

    This is a façade for the methods:
        - soft_sand
        - stiff_sand
        - contact cement
        - constant cement

    Parameters
    ----------
    k : int, float
        Bulk modulus of the mineral.
    g : int, float
        Shear modulus of the mineral.
    phi : int, float, array_like
        Porosity value or log.
    phic : float
        Critical porosity.
    n : int, float
        Coordination number.
    p : int, float
        Hydrostatic confining pressure.
    kc : int, float
        Bulk modulus of the cementing mineral.
    gc : int, float
        Shear modulus of the cementing mineral.
    phib : float
        Porosity where the cement effect starts.

    method : str, optional
        Name of the method to be used. Must be one of
            - 'soft_sand'
            - 'stiff_sand'
            - 'contact cement'
            - 'constant cement ' 
        If not given, default method is 'soft_sand'.

    Returns
    k : array_like
        Bulk modulus using the rock-physics model.
    g : array_like
        Shear modulus using the rock-physics model.

    """
    options = {}

    required = []
    if method == "soft_sand":
        required = ["p"]
    elif method == "stiff_sand":
        required = ["p"]
    elif method == "contact_cement":
        required = ["kc", "gc"]
    elif method == "constant_cement":
        required = ["kc", "gc", "phib"]

    for arg in required:
        if arg not in kwargs:
            msg = f"Missing required argument for method '{method}': '{arg}'"
            raise TypeError(msg)
        options[arg] = kwargs[arg]

    fun = _gem_models[method]

    return fun(k, g, phi, phic, n, **options)


def gem_model(k: float, g: float, phic: float, n: float,
        method: str = "soft_sand", **kwargs) -> np.ndarray:
    """Computes one of the granular effective medium rock-physics models for
    visualization in appy-gui. 

    This is a façade for the methods:
        - soft_sand
        - stiff_sand
        - contact cement
        - constant cement

    Parameters
    ----------
    k : int, float
        Bulk modulus of the mineral.
    g : int, float
        Shear modulus of the mineral.
    phi : int, float, array_like
        Porosity value or log.
    n : int, float
        Coordination number.
    p : int, float
        Hydrostatic confining pressure.
    kc : int, float
        Bulk modulus of the cementing mineral.
    gc : int, float
        Shear modulus of the cementing mineral.
    phib : float
        Porosity where the cement effect starts.

    method : str, optional
        Name of the method to be used. Must be one of
            - 'soft_sand'
            - 'stiff_sand'
            - 'contact cement'
            - 'constant cement ' 
        If not given, default method is 'soft_sand'.

    Returns
    k : array_like
        Bulk modulus using the rock-physics model.
    g : array_like
        Shear modulus using the rock-physics model.

    """
    options = {}

    required = []
    if method == "soft_sand":
        required = ["p"]
    elif method == "stiff_sand":
        required = ["p"]
    elif method == "contact_cement":
        required = ["kc", "gc"]
    elif method == "constant_cement":
        required = ["kc", "gc", "phib"]

    for arg in required:
        if arg not in kwargs:
            msg = f"Missing required argument for method '{method}': '{arg}'"
            raise TypeError(msg)
        options[arg] = kwargs[arg]

    fun = _gem_models[method]

    return fun(k, g, np.linspace(0, phic, 100), phic, n, **options)
