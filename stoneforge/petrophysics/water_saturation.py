import numpy as np
import numpy.typing as npt
import warnings
from stoneforge.petrophysics.helpers import correct_petrophysic_estimation_rage


# Make anomalous water saturation values larger than 1 be one
def correct_range(sw: np.ndarray):
    sw[sw > 1] = 1
    return sw

def archie(rw: float, rt: npt.ArrayLike, phi: npt.ArrayLike, a: float,
           m: float, n: float) -> np.ndarray:
    """Estimate the Water Saturation from Archie's [1]_ equation.

    Parameters
    ----------
    rw : int, float
        Water resistivity.  
    rt : array_like
        Formation resistivity.    
    phi : array_like
        Porosity.   
    a : int, float
        Tortuosity factor.
    m : int, float
        Cementation exponent.
    n : int, float
        Saturation exponent.

    Returns
    -------
    sw : array_like
        Water saturation from Archie equation.

    References
    ----------
    .. [1] Archie GE (1942) The electrical resistivity log as an aid in determining some
    reservoir characteristics. Transactions of the AIME, 146(01), 54-62.

    """
    if any(((a*rw) / (phi**m * rt))**(1/n) > 1):
        warnings.warn(UserWarning("saturation of water must be a value between 0 and 1"))
        return ((a*rw) / (phi**m * rt))**(1/n)

    else:
        sw = ((a*rw) / (phi**m * rt))**(1/n)
        sw = correct_petrophysic_estimation_rage(sw)
        return sw


def simandoux(rw: float, rt: npt.ArrayLike, phi: npt.ArrayLike, a: float,
              m: float, n: float, vsh: npt.ArrayLike,
              rsh: float) -> np.ndarray:
    """Estimate water saturation from Simandoux [1]_ equation.

    Parameters
    ----------
    rw : int, float
        Water resistivity.
    rt : array_like
        True resistivity.    
    phi : array_like
        Porosity.
    a : int, float
        Tortuosity factor.
    m : int, float
        Cementation exponent.
    n : int, float
        Saturation exponent.
    vsh : array_like
        Clay volume log.
    rsh : int, float
        Clay resistivity.

    Returns
    -------
    sw : array_like
        Water saturation from Simandoux equation.

    References
    ----------
    .. [1] Simandoux P (1963) Measures die techniques an milieu application a measure des
    saturation en eau, etude du comportement de massifs agrileux. Review du’Institute Francais
    du Patrole 18(Supplemen-tary Issue):193

    """
    C = (1 - vsh) * a * rw / phi**m
    D = C * vsh / (2*rsh)
    E = C / rt
    sw = ((D**2 + E)**0.5 - D)**(2/n)

    sw = correct_petrophysic_estimation_rage(sw)


    return sw


def indonesia(rw: float, rt: npt.ArrayLike, phi: npt.ArrayLike, a: float,
              m: float, n: float, vsh: npt.ArrayLike,
              rsh: float) -> np.ndarray:
    """Estimate water saturation from Poupon-Leveaux (Indonesia) [1]_ equation.

    Parameters
    ----------
    rw : int, float
        Water resistivity.  
    rt : array_like
        True resistivity.    
    phi : array_like
        Porosity.     
    vsh : array_like
        Clay volume log.
    a : int, float
        Tortuosity factor.
    m : int, float
        Cementation exponent.
    n : int, float
        Saturation exponent.
    rsh : float
        Clay resistivity.

    Returns
    -------
    indonesia : array_like
        Water saturation from Poupon-Leveaux equation.

    References
    ----------
    .. [1] Poupon, A. and Leveaux, J. (1971) Evaluation of Water Saturation in Shaly Formations.
    The Log Analyst, 12, 1-2.

    """
    sw = ((1/rt)**0.5 / ((vsh**(1 - 0.5*vsh) / (rsh)**0.5) + (phi**m / a*rw)**0.5))**(2/n)
    sw = correct_petrophysic_estimation_rage(sw)


    return sw


def fertl(rw: float, rt: npt.ArrayLike, phi: npt.ArrayLike, a: float,
          m: float, vsh: npt.ArrayLike, alpha: float) -> np.ndarray:
    """Estimate water saturation from Fertl [1]_ equation.

    Parameters
    ----------
    rw : int, float
        Water resistivity.
    rt : array_like
        True resistivity.    
    phi : array_like
        Porosity (must be effective).  
    vsh : array_like
        Clay volume log.     
    a : int, float
        Tortuosity factor.
    m : int, float
        Cementation exponent.
    alpha : int, float
        Alpha parameter from Fertl equation.

    Returns
    -------
    fertl : array_like
        Water saturation from Fertl equation.

    References
    ----------
    .. [1] Fertl, W. H. (1975, June). Shaly sand analysis in development wells.
       In SPWLA 16th Annual Logging Symposium. OnePetro.

    """
    sw = phi**(-m/2) * ((a*rw/rt + (alpha*vsh/2)**2)**0.5 - (alpha*vsh/2))
    sw = correct_petrophysic_estimation_rage(sw)


    return sw


_sw_methods = {
    "archie": archie,
    "simandoux": simandoux,
    "indonesia": indonesia,
    "fertl": fertl
}


def water_saturation(rw: float, rt: npt.ArrayLike, phi: npt.ArrayLike,
                     a: float, m: float, method: str = "archie",
                     **kwargs) -> np.ndarray:
    """Compute water saturation from resistivity log.

    This is a façade for the methods:
        - archie
        - simandoux
        - indonesia
        - fertl

    Parameters
    ----------
    rw : int, float
        Water resistivity.
    rt : array_like
        True resistivity.
    phi : array_like
        Porosity (must be effective).
    a : int, float
        Tortuosity factor.
    m : int, float
        Cementation exponent.
    n : int, float
        Saturation exponent. Required if `method` is "archie", "simandoux" or
        "indonesia".
    vsh : array_like
        Clay volume log. Required if `method` is "simandoux", "indonesia" or
        "fertl".
    rsh : float
        Clay resistivity. Required if `method` is "simandoux" or "indonesia".
    alpha : array_like
        Alpha parameter from Fertl equation. Required if `method` is "fertl"
    method : str, optional
        Name of the method to be used.  Should be one of
            - 'archie'
            - 'simandoux'
            - 'indonesia'
            - 'fertl
        If not given, default method is 'archie'

    Returns
    -------
    water_saturation : array_like
        Water saturation for the aimed interval using the defined method.

    """
    options = {}
    
    required = []
    if method == "archie":
        required = ["n"]
    elif method == "simandoux":
        required = ["n", "vsh", "rsh"]
    elif method == "indonesia":
        required = ["n", "vsh", "rsh"]
    elif method == "fertl":
        required = ["vsh", "alpha"]
    
    for arg in required:
        if arg not in kwargs:
            msg = f"Missing required argument for method '{method}': '{arg}'"
            raise TypeError(msg)
        options[arg] = kwargs[arg]
    
    fun = _sw_methods[method]


    sw = fun(rw, rt, phi, a, m, **options)
    
    return sw
