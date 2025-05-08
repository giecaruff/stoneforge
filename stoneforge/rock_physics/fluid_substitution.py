import numpy as np
import numpy.typing as npt


def kdry(phi: npt.ArrayLike, ks: npt.ArrayLike, ksatA: npt.ArrayLike,
         kfluidA: npt.ArrayLike) -> np.ndarray:
    """Calculate the dry-rock bulk modulus using Gassmann' [1]_ equation .

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

    Returns:
    -------
    kdry : array_like
        Dry-rock bulk modulus.

    References:
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    kdry_num = 1 - (1 - phi) * (ksatA/ks) - (phi * ksatA/kfluidA)
    kdry_den = 1 + phi - (phi*ks / kfluidA) - (ksatA/ks)
    kdry = ks * (kdry_num / kdry_den)

    return kdry
    

def ksat(phi: npt.ArrayLike, ks: npt.ArrayLike, kdry: npt.ArrayLike,
         kfluidB: npt.ArrayLike) -> np.ndarray:
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

    Returns:
    -------
    ksat : array_like
        Bulk modulus of the rock saturated with fluid B.

    References:
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    ksat_num = phi*kdry - (1 + phi)*(kfluidB * kdry / ks) + kfluidB
    ksat_den = (1 - phi)*kfluidB + phi*ks - (kfluidB * kdry / ks)
    ksat = ks * (ksat_num / ksat_den)

    return ksat


def gassmann_subs(phi: npt.ArrayLike, ks: npt.ArrayLike, ksatA: npt.ArrayLike,
                  kfluidA: npt.ArrayLike,
                  kfluidB: npt.ArrayLike) -> np.ndarray:
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

    Returns:
    -------
    ksat : array_like
        Bulk modulus of rock saturated with fluid B.

    References:
    ----------
    .. [1] Avseth, Per, Tapan Mukerji, and Gary Mavko. Quantitative seismic
    interpretation: Applying rock physics tools to reduce interpretation risk.
    Cambridge university press, 2005.

    """
    A = ksatA / (ks - ksatA)
    B = kfluidA / (phi*(ks - kfluidA))
    C = kfluidB / (phi*(ks - kfluidB))
    D = A - B + C
    ksat = D*ks / (1 + D)

    return ksat


_gassmann_equations = {
    "kdry": kdry,
    "ksat": ksat,
    "gassmann_subs": gassmann_subs
}


def gassmann(phi: npt.ArrayLike, ks: npt.ArrayLike,
             method: str = "ksat_direct", **kwargs) -> np.ndarray:
    """Compute Gassmann' equations.

    This is a façade for the methods:
        - kdry
        - ksat
        - gassmann_subs

    Parameters
    ----------
    phi : array_like
        Porosity log.
    ks : array_like 
        Bulk modulus of solid phase.
    kdry : array_like
        Bulk modulus of the dry-rock. Required if `method` is ksat.
    ksatA : array_like
        Bulk modulus of the rock saturated with fluid A. Required if method is
        `ksat` or `gassmann_subs`.
    kfluidA : array_like
        Bulk modulus of the fluid A. Required if method is `kdry` or
        `gassmann_subs`.
    kfluidB : array_like 
        Bulk modulus of the fluid B. Required if method is `ksat` or
        `gassmann_subs`.
    method : str, optional
        Name of the method to be used. Should be one of
            - 'kdry'
            - 'ksat'
            - 'ksat_direct'
        If not given, default method is 'gassmann_subs'.

    Returns:
    -------
    ksat : array_like
        Bulk modulus of the rock saturated with fluid B.
    
    """
    options = {}

    required = []
    if method == "kdry":
        required = ["ksatA", "kfluidA"]
    elif method == "ksat":
        required = ["kdry", "kfluidB"]
    elif method == "gassmann_subs":
        required = ["ksatA", "kfluidA", "kfluidB"]

    for arg in required:
        if arg not in kwargs:
            msg = f"Missing required argument for method '{method}': '{arg}'"
            raise TypeError(msg)
        options[arg] = kwargs[arg]

    fun = _gassmann_equations[method]

    return fun(phi, ks, **options)


def mdry(phi: npt.ArrayLike, ms: npt.ArrayLike, msatA: npt.ArrayLike,
         kfluidA: npt.ArrayLike) -> np.ndarray:
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

    Returns:
    -------
    mdry : array_like
        Dry-rock compressional modulus.

    References:
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    mdry_num = 1 - (1 - phi) * (msatA/ms) - (phi * msatA/kfluidA)
    mdry_den = 1 + phi - (phi * ms/kfluidA) - (msatA/ms)
    mdry = ms * (mdry_num / mdry_den)

    return mdry


def msat(phi: npt.ArrayLike, ms: npt.ArrayLike, mdry: npt.ArrayLike,
         kfluidB: npt.ArrayLike) -> np.ndarray:
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

    Returns:
    -------
    msat : array_like
        Compressional modulus of the rock saturated with fluid B.

    References:
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    msat_num = phi*mdry - (1 + phi) * (kfluidB * mdry/ms) + kfluidB
    msat_den = (1 - phi) * kfluidB + (phi*ms) - (kfluidB * mdry/ms)
    msat = ms * (msat_num / msat_den)

    return msat


def mavko_subs(phi: npt.ArrayLike, ms: npt.ArrayLike, msatA: npt.ArrayLike,
                  kfluidA: npt.ArrayLike,
                  kfluidB: npt.ArrayLike) -> np.ndarray:
    """Fluid substitution using Mavko' equation without calculating
    dry-rock bulk modulus [1]_.

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

    Returns:
    -------
    msat : array_like
        Compressional modulus of rock saturated with fluid B.

    References:
    ----------
    .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
    properties. [S.l.]: Cambridge University Press, 2014.

    """
    A = msatA / (ms - msatA)
    B = kfluidA / (phi*(ms - kfluidA))
    C = kfluidB / (phi*(ms - kfluidB))
    D = A - B + C
    msat = D*ms / (1 + D)

    return msat


_mavko_equations = {
    "mdry": mdry,
    "msat": msat,
    "mavko_subs": mavko_subs
}


def mavko(phi: npt.ArrayLike, ms: npt.ArrayLike,
             method: str = "msat_direct", **kwargs) -> np.ndarray:
    """Compute Mavko' equations.

    This is a façade for the methods:
        - mdry
        - msat
        - mavko_subs

    Parameters
    ----------
    phi : array_like
        Porosity log.
    ms : array_like 
        Compressional modulus of solid phase.
    mdry : array_like
        Compressional modulus of the dry-rock. Required if `method` is msat.
    msatA : array_like
        Compressional modulus of the rock saturated with fluid A. Required if method is
        `msat` or `mavko_subs`.
    kfluidA : array_like
        Bulk modulus of the fluid A. Required if method is `mdry` or
        `mavko_subs`.
    kfluidB : array_like 
        Bulk modulus of the fluid B. Required if method is `msat` or
        `mavko_subs`.
    method : str, optional
        Name of the method to be used. Should be one of
            - 'mdry'
            - 'msat'
            - 'msat_direct'
        If not given, default method is 'mavko_subs'.

    Returns:
    -------
    msat : array_like
        Compressional modulus of the rock saturated with fluid B.
    
    """
    options = {}

    required = []
    if method == "mdry":
        required = ["msatA", "kfluidA"]
    elif method == "msat":
        required = ["mdry", "kfluidB"]
    elif method == "mavko_subs":
        required = ["msatA", "kfluidA", "kfluidB"]

    for arg in required:
        if arg not in kwargs:
            msg = f"Missing required argument for method '{method}': '{arg}'"
            raise TypeError(msg)
        options[arg] = kwargs[arg]

    fun = _mavko_equations[method]

    return fun(phi, ms, **options)
