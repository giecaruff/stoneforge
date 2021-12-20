from numpy import clip
#import pytest


def archie(rw,rt,phi,a,m,n):
    """Estimate the Water Saturation from Archie [1]_ equation.

    Parameters
    ----------
    rw : array_like
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

    sw_archie = ((a*rw)/(phi**m * rt))**(1./n)
    sw_archie = clip(sw_archie, 0., 1.)

    return sw_archie


def simandoux(rw, rt, phi, a, m, n, vsh, rsh):
    """Estimate water saturation from Simandoux [1]_ equation.

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
        Saturation exponent.
    vsh : array_like
        Clay volume log.
    rsh : float
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

    sw_simandoux = ((a*rw / rt*(phi**m)) + (a*rw/(phi**m) * vsh/2*rsh)**2)**(1/n) - (a*rw/(phi**m) * vsh/2*rsh)
    sw_simandoux = clip(sw_simandoux, 0., 1.)

    return sw_simandoux


def indonesia(rw, rt, phi, a, m, n, vsh, rsh):
    """Estimate water saturation from Poupon-Leveaux (Indonesia) [1]_ equation.

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
        Saturation exponent.
    vsh : array_like
        Clay volume log.
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

    sw_indonesia = ((1/rt)**0.5 / ((vsh**(1 - 0.5*vsh) / (rsh)**0.5) + (phi**m / a*rw)**0.5))**(2/n)
    sw_indonesia = clip(sw_indonesia, 0., 1.)

    return sw_indonesia


def fertl(rw, rt, phi, a, m, vsh, alpha):
    """Estimate water saturation from Fertl [1]_ equation.

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
    vsh : array_like
        Clay volume log.
    alpha : array_like
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

    sw_fertl = phi**(-m/2) * ((a*rw/rt + (alpha*vsh/2)**2)**0.5 - (alpha*vsh/2))
    sw_fertl = clip(sw_fertl, 0., 1.)

    return sw_fertl


_sw_methods = {
    "archie": archie,
    "simandoux": simandoux,
    "indonesia": indonesia,
    "fertl": fertl
}

def water_saturation(rw, rt, phi, a, m, method="archie", **kwargs):
    """Compute water saturation from resistivity log.

    This is a facade for the methods:
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
        Water saturation for the aimed interval using the specific method.
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

    return fun(rw, rt, phi, a, m, **options)
