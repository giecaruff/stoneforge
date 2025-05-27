# -*- coding: utf-8 -*-

import numpy as np
from typing import Annotated


def kdry(phi: Annotated[np.array, "Porosity"],
         ks: Annotated[np.array, "Bulk modulus of solid phase"],
         ksatA: Annotated[np.array, "Bulk modulus for fluid A"],
         kfluidA: Annotated[np.array, "Bulk modulus of the fluid A"]) -> np.array:
    """Calculate the dry-rock bulk modulus using Gassmann' [1]_ equation .

    Parameters
    ----------
    phi : array_like
        Porosity information.

    ks : array_like
        Bulk modulus of solid phase.

    ksatA : array_like
        Bulk modulus of the rock phase saturated with fluid A.

    kfluidA : array_like
        Bulk modulus of the fluid A.

    Returns
    -------
    kdry : array_like
        Dry-rock bulk modulus.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock properties. [S.l.]: Cambridge University Press, 2014.
    """
    kdry_num = 1 - (1 - phi) * (ksatA/ks) - (phi * ksatA/kfluidA)
    kdry_den = 1 + phi - (phi*ks / kfluidA) - (ksatA/ks)

    return ks * (kdry_num / kdry_den)


def ksat(phi: Annotated[np.array, "Porosity"],
         ks: Annotated[np.array, "Bulk modulus of solid phase"],
         kdry: Annotated[np.array, "Dry-rock bulk modulus"],
         kfluidB: Annotated[np.array, "Bulk modulus of the fluid B"]) -> np.array:
    """Calculate the bulk modulus of the rock saturated with fluid B [1]_.

    Parameters
    ----------
    phi : array_like
        Porosity log.

    ks : array_like
        Bulk modulus of solid phase.

    kdry : array_like
        Bulk modulus of the dry-rock.

    kfluidB : array_like
        Bulk modulus of the fluid B.

    Returns
    -------
    ksat : array_like
        Bulk modulus of the rock saturated with fluid B.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock properties. [S.l.]: Cambridge University Press, 2014.
    """
    ksat_num = phi*kdry - (1 + phi)*(kfluidB * kdry / ks) + kfluidB
    ksat_den = (1 - phi)*kfluidB + phi*ks - (kfluidB * kdry / ks)

    return ks * (ksat_num / ksat_den)


def gassmann_subs(phi: Annotated[np.array, "Porosity"],
                  ks: Annotated[np.array, "Bulk modulus of solid phase"],
                  ksatA: Annotated[np.array, "Bulk modulus for fluid A"],
                  kfluidA: Annotated[np.array, "Bulk modulus of the fluid A"],
                  kfluidB: Annotated[np.array, "Bulk modulus of the fluid B"]) -> np.array:
    """Fluid substitution using Gassmann' equation without calculating
    dry-rock bulk modulus [1]_.

    Parameters
    ----------
    phi : array_like
        Porosity log.

    ks : array_like
        Bulk modulus of solid phase.

    ksatA : array_like
        Bulk modulus of the rock saturated with fluid A.

    kfluidA : array_like
        Bulk modulus of the fluid A.

    kfluidB : array_like
        Bulk modulus of the fluid B.

    Returns
    -------
    ksat : array_like
        Bulk modulus of rock saturated with fluid B.

    References
    ----------
    .. [1] Avseth, Per, Tapan Mukerji, and Gary Mavko. Quantitative seismic interpretation: Applying rock physics tools to reduce interpretation risk. Cambridge university press, 2005.
    """
    A = ksatA / (ks - ksatA)
    B = kfluidA / (phi*(ks - kfluidA))
    C = kfluidB / (phi*(ks - kfluidB))
    D = A - B + C

    return D*ks / (1 + D)

#_gassmann_equations = {
#    "kdry": kdry,
#    "ksat": ksat,
#    "gassmann_subs": gassmann_subs
#}

def gassmann(phi: Annotated[np.array, "Porosity"],
             ks: Annotated[np.array, "Bulk modulus of solid phase"],
             method: Annotated[str, "Chosen method for Gassmann fluid substitution"] = 'gassmann_subs', **kwargs) -> np.array:
    """
    Compute Gassmann fluid substitution and modulus equations.

    This function serves as a façade for different Gassmann-related methods:
        - Dry-rock bulk modulus estimation (`kdry`)
        - Saturated-rock bulk modulus estimation (`ksat`)
        - Gassmann fluid substitution (`gassmann_subs`)

    Parameters
    ----------
    phi : array_like
        Porosity of the rock [fractional, e.g., 0.25 for 25%].
    
    ks : array_like
        Bulk modulus of the solid grain matrix [GPa or consistent unit].

    method : {'kdry', 'ksat', 'gassmann_subs'}, default='gassmann_subs'
        The Gassmann equation variant to apply.
        - 'kdry'           : Compute dry rock bulk modulus from known saturated rock.
        - 'ksat'           : Compute saturated rock bulk modulus from dry rock.
        - 'gassmann_subs'  : Substitute fluid in the saturated rock.

    ksatA : array_like, optional
        Saturated bulk modulus with fluid A [GPa]. Required for 'kdry' and 'gassmann_subs'.
    
    kfluidA : array_like, optional
        Bulk modulus of fluid A [GPa]. Required for 'kdry' and 'gassmann_subs'.

    kfluidB : array_like, optional
        Bulk modulus of fluid B [GPa]. Required for 'ksat' and 'gassmann_subs'.
    
    kdry : array_like, optional
        Dry rock bulk modulus [GPa]. Required for 'ksat'.

    Returns
    -------
    ksat : array_like
        Bulk modulus of rock saturated with fluid B [GPa].

    Raises
    ------
    TypeError
        If required parameters for the selected method are missing.

    ValueError
        If an unsupported method is specified.

    References
    ----------
    .. [1] Gassmann, F. (1951). Elastic waves through a packing of spheres.
           *Geophysics*, 16(4), 673-685.

    .. [2] Mavko, G., Mukerji, T., & Dvorkin, J. (2009). *The Rock Physics Handbook*.
           Cambridge University Press.

    Examples
    --------
    >>> gassmann(phi=0.25, ks=36, method="kdry", ksatA=25, kfluidA=2.2)
    array([...])

    >>> gassmann(phi=0.25, ks=36, method="gassmann_subs", ksatA=25, kfluidA=2.2, kfluidB=1.5)
    array([...])
    """
    method_map = {
        "kdry": (kdry, ["ksatA", "kfluidA"]),
        "ksat": (ksat, ["kdry", "kfluidB"]),
        "gassmann_subs": (gassmann_subs, ["ksatA", "kfluidA", "kfluidB"])
    }

    if method not in method_map:
        raise ValueError(f"Unsupported method '{method}'. Choose from: {list(method_map.keys())}")

    func, required_args = method_map[method]
    if missing := [arg for arg in required_args if arg not in kwargs]:
        raise TypeError(f"Missing required arguments for method '{method}': {', '.join(missing)}")

    # Collect required arguments for the selected method
    method_args = {key: kwargs[key] for key in required_args}

    return func(phi, ks, **method_args)




def mdry(phi: Annotated[np.array, "Porosity"],
         ms: Annotated[np.array, "Compressional modulus of solid phase"],
         msatA: Annotated[np.array, "Compressional modulus of the rock saturated with fluid A"],
         kfluidA: Annotated[np.array, "Bulk modulus of the fluid A"]) -> np.array:
    """Calculate the dry-rock compressional modulus using Mavko' [1]_ equation .

    Parameters
    ----------
    phi : array_like
        Porosity log.

    ms : array_like
        Compressional modulus of solid phase.

    msatA : array_like
        Compressional modulus of the rock saturated with fluid A.

    kfluidA : array_like
        Bulk modulus of the fluid A.

    Returns
    -------
    mdry : array_like
        Dry-rock compressional modulus.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock properties. [S.l.]: Cambridge University Press, 2014.
    """
    mdry_num = 1 - (1 - phi) * (msatA/ms) - (phi * msatA/kfluidA)
    mdry_den = 1 + phi - (phi * ms/kfluidA) - (msatA/ms)

    return ms * (mdry_num / mdry_den)



def msat(phi: Annotated[np.array, "Porosity"],
         ms: Annotated[np.array, "Compressional modulus of solid phase"],
         mdry: Annotated[np.array, "Compressional modulus of the dry-rock"],
         kfluidB: Annotated[np.array, "Bulk modulus of the fluid B"]) -> np.array:
    """Calculate the compressional modulus of the rock saturated with fluid B [1]_.

    Parameters
    ----------
    phi : array_like
        Porosity log.

    ms : array_like
        Compressional modulus of solid phase.

    mdry : array_like
        Compressional modulus of the dry-rock.

    kfluidB : array_like
        Bulk modulus of the fluid B.

    Returns
    -------
    msat : array_like
        Compressional modulus of the rock saturated with fluid B.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock properties. [S.l.]: Cambridge University Press, 2014.
    """
    msat_num = phi*mdry - (1 + phi) * (kfluidB * mdry/ms) + kfluidB
    msat_den = (1 - phi) * kfluidB + (phi*ms) - (kfluidB * mdry/ms)

    return ms * (msat_num / msat_den)


def mavko_subs(phi: Annotated[np.array, "Porosity"],
               ms: Annotated[np.array, "Compressional modulus of solid phase"],
               msatA: Annotated[np.array, "Compressional modulus of the rock saturated with fluid A"],
               kfluidA: Annotated[np.array, "Bulk modulus of the fluid A"],
               kfluidB: Annotated[np.array, "Bulk modulus of the fluid B"]) -> np.array:
    """Fluid substitution using Mavko' equation without calculating dry-rock bulk modulus [1]_.

    Parameters
    ----------
    phi : array_like
        Porosity log.

    ms : array_like
        Compressional modulus of solid phase.

    msatA : array_like
        Compressional modulus of the rock saturated with fluid A.

    kfluidA : array_like
        Bulk modulus of the fluid A.

    kfluidB : array_like
        Bulk modulus of the fluid B.

    Returns
    -------
    msat : array_like
        Compressional modulus of rock saturated with fluid B.

    References
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock properties. [S.l.]: Cambridge University Press, 2014.
    """
    A = msatA / (ms - msatA)
    B = kfluidA / (phi*(ms - kfluidA))
    C = kfluidB / (phi*(ms - kfluidB))
    D = A - B + C

    return D*ms / (1 + D)


#_mavko_equations = {
#    "mdry": mdry,
#    "msat": msat,
#    "mavko_subs": mavko_subs
#}


def mavko(
    phi: Annotated[np.array, "Porosity of the rock (fractional)"],
    ms: Annotated[np.array, "Compressional modulus of the solid matrix (Ms) [GPa]"],
    method: Annotated[str, "Chosen method for Gassmann fluid substitution"] = "mavko_subs",
    **kwargs) -> np.array:
    """
    Compute Mavko's fluid substitution and modulus equations.

    This is a façade for different Mavko equation implementations:
        - Dry-rock compressional modulus (`mdry`)
        - Saturated-rock compressional modulus (`msat`)
        - Mavko fluid substitution (`mavko_subs`)

    Parameters
    ----------
    phi : array_like
        Porosity of the rock [fractional, e.g., 0.25 for 25%].
    
    ms : array_like
        Compressional modulus of the solid phase [GPa].

    method : {'mdry', 'msat', 'mavko_subs'}, default='mavko_subs'
        The Mavko equation variant to apply.
        - 'mdry'         : Estimate dry rock modulus from saturated rock.
        - 'msat'         : Estimate saturated rock modulus from dry rock.
        - 'mavko_subs'   : Perform fluid substitution using two fluids.

    msatA : array_like, optional
        Compressional modulus of rock saturated with fluid A [GPa]. Required for 'mdry' and 'mavko_subs'.

    kfluidA : array_like, optional
        Bulk modulus of fluid A [GPa]. Required for 'mdry' and 'mavko_subs'.

    kfluidB : array_like, optional
        Bulk modulus of fluid B [GPa]. Required for 'msat' and 'mavko_subs'.

    mdry : array_like, optional
        Compressional modulus of dry rock [GPa]. Required for 'msat'.

    Returns
    -------
    msat : array_like
        Compressional modulus of rock saturated with fluid B [GPa].

    Raises
    ------
    TypeError
        If required parameters for the selected method are missing.

    ValueError
        If an unsupported method is provided.

    References
    ----------
    .. [1] Mavko, G., Mukerji, T., & Dvorkin, J. (2009).
           *The Rock Physics Handbook*. Cambridge University Press.

    Examples
    --------
    >>> mavko(phi=0.25, ms=39, method="mdry", msatA=32, kfluidA=2.2)
    array([...])

    >>> mavko(phi=0.25, ms=39, method="mavko_subs", msatA=32, kfluidA=2.2, kfluidB=1.5)
    array([...])
    """
    method_map = {
        "mdry": (mdry, ["msatA", "kfluidA"]),
        "msat": (msat, ["mdry", "kfluidB"]),
        "mavko_subs": (mavko_subs, ["msatA", "kfluidA", "kfluidB"]),
    }

    if method not in method_map:
        raise ValueError(f"Unsupported method '{method}'. Choose from: {list(method_map.keys())}")

    func, required_args = method_map[method]

    if missing := [arg for arg in required_args if arg not in kwargs]:
         raise TypeError(f"Missing required arguments for method '{method}': {', '.join(missing)}")

    method_args = {key: kwargs[key] for key in required_args}

    return func(phi, ms, **method_args)