# -*- coding: utf-8 -*-

import numpy as np
from typing import Annotated
from .elastic_constants import poisson


def hertz_mindlin(k: Annotated[float, "Bulk modulus of the mineral"],
                  g: Annotated[float, "Shear modulus of the mineral"],
                  n: Annotated[float, "Coordination number"],
                  phic: Annotated[float, "Critical porosity"],
                  p: Annotated[float, "Hydrostatic confining pressure"]) -> float:
    """Computes the elastic moduli of the original room-dry grain pack at critical porosity phic from Hertz-Mindlin contact theory [1]_.

    Parameters
    ----------
    k : float
        Bulk modulus of the mineral.
    g : float
        Shear modulus of the mineral.
    n : float
        Coordination number.
    phic : float
        Critical porosity.
    p : float
        Hydrostatic confining pressure.

    Returns
    -------
    khm : float
        Bulk modulus of the Hertz-Mindlin point.
    ghm : float
        Shear modulus of the Hertz-Mindlin point.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock properties. [S.l.]: Cambridge University Press, 2014.
    """
    v = poisson(method = "k_and_g", k=k, g=g)
    khm = ((n**2 * (1.-phic)**2 * g**2 * p) / \
          ( 18*np.pi**2 * (1-v)**2))**(1/3)
    ghm = ((5-4*v)/(5*(2-v))) * ((3*n**2 * (1-phic)**2 * g**2 * p) / \
          ( 2*np.pi**2 * (1-v)**2))**(1/3)

    return khm, ghm


def soft_sand(k: Annotated[float, "Bulk modulus of the mineral"],
              g: Annotated[float, "Shear modulus of the mineral"],
              phi: Annotated[np.array, "Porosity value or log"],
              phic: Annotated[float, "Critical porosity"],
              n: Annotated[float, "Coordination number"],
              p: Annotated[float, "Hydrostatic confining pressure"]) -> np.array:
    """Computes the elastic moduli of the rock using the soft sand model [1]_.

    Parameters
    ----------
    k : float
        Bulk modulus of the mineral.
    g : float
        Shear modulus of the mineral.
    phi : array_like
        Porosity value or log.
    phic : float
        Critical porosity.
    n : float
        Coordination number.
    p : float
        Hydrostatic confining pressure.

    Returns
    -------
    ksoft : array_like
        Bulk modulus using the soft sand model.
    gsoft : array_like
        Shear modulus using the soft sand model.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock properties. [S.l.]: Cambridge University Press, 2014.
    """
    khm, ghm = hertz_mindlin(k, g, n, phic, p)
    zhm = (ghm/6) * (9*khm + 8*ghm)/(khm + 2*ghm)
    ksoft = ((phi/phic)/(khm + 4/3* ghm) + (1 - phi/phic) / \
            (k + 4/3*ghm))**-1 - 4/3 * ghm
    gsoft = ((phi/phic)/(ghm + zhm) + (1 - phi/phic)/(g + zhm))**-1 - zhm
    
    return ksoft, gsoft


def constant_cement(k: Annotated[float, "Bulk modulus of the mineral"],
                    g: Annotated[float, "Shear modulus of the mineral"],
                    phi: Annotated[np.array, "Porosity value or log"],
                    phic: Annotated[float, "Critical porosity"],
                    n: Annotated[float, "Coordination number"],
                    kc: Annotated[float, "Bulk modulus of the cementing mineral"],
                    gc: Annotated[float, "Shear modulus of the cementing mineral"],
                    phib: Annotated[float, "Porosity where the cement effect starts"],
                    deposition_type: Annotated[str, "Cement deposition framework: grain_surface or grain_contact"] = 'grain_surface') -> np.array:
    """Computes the elastic moduli of the rock using the constant cement model [1]_.

    Parameters
    ----------
    k : float
        Bulk modulus of the mineral.
    g : float
        Shear modulus of the mineral.
    phi : array_like
        Porosity value or log.
    phic : float
        Critical porosity.
    n : float
        Coordination number.
    kc : float
        Bulk modulus of the cementing mineral.
    gc : float
        Shear modulus of the cementing mineral.
    phib : float
        Porosity where the cement effect starts.
    deposition_type : str
        Cement deposition framework: grain_surface or grain_contact

    Returns
    -------
    kconst : array_like
        Bulk modulus using the constant cement model.
    gconst : array_like
        Shear modulus using the constant cement model.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock properties. [S.l.]: Cambridge University Press, 2014.
    """
    if isinstance(phi, float):
        kconst, gconst = np.zeros((1)), np.zeros((1))
    else:
        kconst, gconst = np.zeros(phi.shape), np.zeros(phi.shape)
    
    soft_domain = phi < phib
    cement_domain = phi >= phib

    kcem, gcem = contact_cement(k, g, phi, phic, n, kc, gc, deposition_type)
    kend, gend = contact_cement(k, g, phib, phic, n, kc, gc, deposition_type)
    
    if not isinstance(phi, float):
        kconst[cement_domain], gconst[cement_domain] = kcem[cement_domain], gcem[cement_domain]
        
        kb, gb = kend, gend
        zb = (gb/6) * (9*kb + 8*gb)/(kb + 2*gb)
        kco = ((phi/phib)/(kb + 4/3* gb) + (1 - phi/phib)/(k + 4/3*gb))**-1 - 4/3 * gb
        gco = ((phi/phib)/(gb + zb) + (1 - phi/phib)/(g + zb))**-1 - zb
        kconst[soft_domain], gconst[soft_domain] = kco[soft_domain], gco[soft_domain]    
    
    elif cement_domain:
        kconst, gconst = kcem, gcem
    else:
        kb, gb = kend, gend
        zb = (gb/6) * (9*kb + 8*gb)/(kb + 2*gb)
        kco = ((phi/phib)/(kb + 4/3* gb) + (1 - phi/phib)/(k + 4/3*gb))**-1 - 4/3 * gb
        gco = ((phi/phib)/(gb + zb) + (1 - phi/phib)/(g + zb))**-1 - zb
        kconst, gconst = kco, gco

    return kconst, gconst


def stiff_sand(k: Annotated[float, "Bulk modulus of the mineral"],
               g: Annotated[float, "Shear modulus of the mineral"],
               phi: Annotated[np.array, "Porosity value or log"],
               phic: Annotated[float, "Critical porosity"],
               n: Annotated[float, "Coordination number"],
               p: Annotated[float, "Hydrostatic confining pressure"]) -> np.array:
    """Computes the elastic moduli of the rock using the stiff sand model [1]_.

    Parameters
    ----------
    k : float
        Bulk modulus of the mineral.
    g : float
        Shear modulus of the mineral.
    phi : array_like
        Porosity value or log.
    phic : float
        Critical porosity.
    n : float
        Coordination number.
    p : float
        Hydrostatic confining pressure.

    Returns
    -------
    kstiff : array_like
        Bulk modulus using the stiff sand model.
    gstiff : array_like
        Shear modulus using the stiff sand model.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock properties. [S.l.]: Cambridge University Press, 2014.
    """
    khm, ghm = hertz_mindlin(k, g, n, phic, p)
    z = (g/6) * (9*k + 8*g)/(k + 2*g)
    kstiff = ((phi/phic)/(khm + 4/3* g) + (1 - phi/phic)/(k + 4/3*g))**-1 - 4/3 * g
    gstiff = ((phi/phic)/(ghm + z) + (1 - phi/phic)/(g + z))**-1 - z
    
    return kstiff, gstiff


def contact_cement(k: Annotated[float, "Bulk modulus of the mineral"],
                   g: Annotated[float, "Shear modulus of the mineral"],
                   phi: Annotated[np.array, "Porosity value or log"],
                   phic: Annotated[float, "Critical porosity"],
                   n: Annotated[float, "Coordination number"],
                   kc: Annotated[float, "Bulk modulus of the cementing mineral"],
                   gc: Annotated[float, "Shear modulus of the cementing mineral"],
                   deposition_type: Annotated[str, "Cement deposition framework: grain_surface or grain_contact"] = 'grain_surface') -> np.array:
    """Computes the elastic moduli of the rock using the contact cement model [1]_.

    Parameters
    ----------
    k : float
        Bulk modulus of the mineral.
    g : float
        Shear modulus of the mineral.
    phi : array_like
        Porosity value or log.
    phic : float
        Critical porosity.
    n : float
        Coordination number.
    kc : float
        Bulk modulus of the cementing mineral.
    gc : float
        Shear modulus of the cementing mineral.
    deposition_type : str
        Cement deposition framework: grain_surface or grain_contact

    Returns
    -------
    kcem : array_like
        Bulk modulus using the contact cement model.
    gcem : array_like
        Shear modulus using the contact cement model.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock properties. [S.l.]: Cambridge University Press, 2014.
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


def gem(
    k: Annotated[float, "Bulk modulus of the mineral phase [GPa]"],
    g: Annotated[float, "Shear modulus of the mineral phase [GPa]"],
    phi: Annotated[np.array, "Porosity (scalar or array) [fractional]"],
    phic: Annotated[float, "Critical porosity [fractional]"],
    n: Annotated[float, "Coordination number (typically 6–10)"],
    method: Annotated[str, "Granular effective medium model"] = "soft_sand",
    **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the elastic moduli using granular effective medium (GEM) rock physics models.

    This function wraps several GEM-based models including:
        - Soft sand model
        - Stiff sand model
        - Contact cement model
        - Constant cement model

    Parameters
    ----------
    k : float
        Bulk modulus of the mineral frame [GPa].

    g : float
        Shear modulus of the mineral frame [GPa].

    phi : array_like
        Porosity log or scalar value [fractional].

    phic : float
        Critical porosity [fractional].

    n : float
        Coordination number (average number of grain contacts).

    method : {'soft_sand', 'stiff_sand', 'contact_cement', 'constant_cement'}, default='soft_sand'
        The granular effective medium model to apply.

    p : float, (optional)
        Confining pressure [MPa]. Required for 'soft_sand' and 'stiff_sand'.

    kc : float, (optional)
        Bulk modulus of cement [GPa]. Required for 'contact_cement' and 'constant_cement'.

    gc : float, (optional)
        Shear modulus of cement [GPa]. Required for 'contact_cement' and 'constant_cement'.

    phib : float, (optional)
        Porosity at the beginning of cementation [fractional]. Required for 'constant_cement'.

    Returns
    -------
    k_model : array_like
        Bulk modulus computed from the chosen model [GPa].

    g_model : array_like
        Shear modulus computed from the chosen model [GPa].

    Raises
    ------
    TypeError
        If required parameters for the selected model are missing.

    ValueError
        If an unsupported method is specified.

    References
    ----------
    .. [1] Dvorkin, J., Mavko, G., & Nur, A. (1999). The effect of cementation on the elastic properties of granular material. *Mechanics of Materials*, 12(3), 207–217.

    Examples
    --------
    >>> gem(k=36, g=45, phi=0.25, phic=0.4, n=8, method="soft_sand", p=20)
    (array([...]), array([...]))

    >>> gem(k=36, g=45, phi=0.25, phic=0.4, n=8, method="constant_cement", kc=20, gc=25, phib=0.3)
    (array([...]), array([...]))
    """
    method_map = {
        "soft_sand": (soft_sand, ["p"]),
        "stiff_sand": (stiff_sand, ["p"]),
        "contact_cement": (contact_cement, ["kc", "gc"]),
        "constant_cement": (constant_cement, ["kc", "gc", "phib"])
    }

    if method not in method_map:
        raise ValueError(f"Unsupported method '{method}'. Choose from: {list(method_map.keys())}")

    func, required_args = method_map[method]

    if missing := [arg for arg in required_args if arg not in kwargs]:
         raise TypeError(f"Missing required arguments for method '{method}': {', '.join(missing)}")

    method_args = {key: kwargs[key] for key in required_args}

    return func(k, g, phi, phic, n, **method_args)


def gem_model(
    k: Annotated[float, "Bulk modulus of the mineral phase [GPa]"],
    g: Annotated[float, "Shear modulus of the mineral phase [GPa]"],
    phic: Annotated[float, "Critical porosity [fractional]"],
    n: Annotated[float, "Coordination number (typically 6–10)"],
    method: Annotated[str, "Granular effective medium model"] = "soft_sand",
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute bulk and shear moduli from granular effective medium models for visualization.

    This façade function wraps four common GEM rock-physics models:
        - Soft sand
        - Stiff sand
        - Contact cement
        - Constant cement

    A porosity array from 0 to critical porosity (phic) is used as input.

    Parameters
    ----------
    k : float
        Bulk modulus of the mineral phase [GPa].

    g : float
        Shear modulus of the mineral phase [GPa].

    phic : float
        Critical porosity [fractional].

    n : float
        Coordination number (typically 6–10).

    method : {'soft_sand', 'stiff_sand', 'contact_cement', 'constant_cement'}, default='soft_sand'
        The GEM model to use.

    p : float, optional
        Confining pressure [MPa]. Required for 'soft_sand' and 'stiff_sand'.

    kc : float, optional
        Bulk modulus of the cement [GPa]. Required for 'contact_cement' and 'constant_cement'.

    gc : float, optional
        Shear modulus of the cement [GPa]. Required for 'contact_cement' and 'constant_cement'.

    phib : float, optional
        Porosity where cementation begins [fractional]. Required for 'constant_cement'.

    Returns
    -------
    k_model : np.ndarray
        Bulk modulus curve [GPa].

    g_model : np.ndarray
        Shear modulus curve [GPa].

    Raises
    ------
    TypeError
        If any required parameter for the selected method is missing.

    ValueError
        If the selected method is invalid.

    Examples
    --------
    >>> gem_model(k=36, g=45, phic=0.4, n=8, method="soft_sand", p=25)
    (array([...]), array([...]))

    >>> gem_model(k=36, g=45, phic=0.4, n=8, method="constant_cement", kc=25, gc=30, phib=0.28)
    (array([...]), array([...]))
    """
    method_map = {
        "soft_sand": (soft_sand, ["p"]),
        "stiff_sand": (stiff_sand, ["p"]),
        "contact_cement": (contact_cement, ["kc", "gc"]),
        "constant_cement": (constant_cement, ["kc", "gc", "phib"])
    }

    if method not in method_map:
        raise ValueError(f"Unsupported method '{method}'. Choose from: {list(method_map)}")

    func, required_args = method_map[method]

    if missing := [arg for arg in required_args if arg not in kwargs]:
         raise TypeError(f"Missing required arguments for method '{method}': {', '.join(missing)}")

    # Build porosity curve from 0 to critical porosity
    phi = np.linspace(0, phic, 100)

    # Extract arguments
    method_args = {arg: kwargs[arg] for arg in required_args}

    return func(k, g, phi, phic, n, **method_args)