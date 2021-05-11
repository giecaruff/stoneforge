from numpy import clip
#import pytest


def archie(rw,rt,phi,a,m,n):
    """Estimate the Water Saturation from Archie equation [1].

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

    [1] Archie GE (1942) The electrical resistivity log as an aid in determining some
    reservoir characteristics. Transactions of the AIME, 146(01), 54-62.
    """

    sw_archie = ((a*rw)/(phi**m * rt))**(1./n)
    sw_archie = clip(sw_archie, 0., 1.)

    return sw_archie


def simandoux(rw, rt, phi, a, m, n, vsh, rsh):
    """Estimate water saturation from Simandoux equation [1].

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

    [1] Simandoux P (1963) Measures die techniques an milieu application a measure des
    saturation en eau, etude du comportement de massifs agrileux. Review du’Institute Francais
    du Patrole 18(Supplemen-tary Issue):193
    """

    sw_simandoux = ((a*rw / rt*(phi**m)) + (a*rw/(phi**m) * vsh/2*rsh)**2)**(1/n) - (a*rw/(phi**m) * vsh/2*rsh)
    sw_simandoux = clip(sw_simandoux, 0., 1.)

    return sw_simandoux


def indonesia(rw, rt, phi, a, m, n, vsh, rsh):
    """Estimate water saturation from Poupon-Leveaux (Indonesia) equation [1].

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

    [1] Poupon, A. and Leveaux, J. (1971) Evaluation of Water Saturation in Shaly Formations.
    The Log Analyst, 12, 1-2.
    """

    sw_indonesia = ((1/rt)**0.5 / ((vsh**(1 - 0.5*vsh) / (rsh)**0.5) + (phi**m / a*rw)**0.5))**(2/n)
    sw_indonesia = clip(sw_indonesia, 0., 1.)

    return sw_indonesia


def fertl(rw, rt, phi, a, m, vsh, alpha):
    """Estimate water saturation from Fertl equation [1].

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
    [1] Fertl, W. H. (1975, June). Shaly sand analysis in development wells. In SPWLA 16th
    Annual Logging Symposium. OnePetro.
    """

    sw_fertl = phi**(-m/2) * ((a*rw/rt + (alpha*vsh/2)**2)**0.5 - (alpha*vsh/2))
    sw_fertl = clip(sw_fertl, 0., 1.)

    return sw_fertl
