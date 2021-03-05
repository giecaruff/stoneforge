import logging
import pytest

    
def velocity_si(sonic_transit):

    """Convert the velocity to International System (I.S.).

    Parameters
    ----------
    sonic_transit : int, float, array_like
        sonic_transit values in standard us/ft log type.

    Returns
    -------
    velocity : int, float, array_like
        velocity in International System (m/s).
    """

    velocity = 304800/sonic_transit

    return velocity


def vagarosity_sl(velocity):

    """Convert the sonic transit time to standard log system (us/ft) .

    Parameters
    ----------
    velocity : int, float, array_like
        velocity values in International System (m/s).

    Returns
    -------
    sonic_transit : int, float, array_like
        sonic_transit in International System (us/ft).
    """

    sonic_transit = 304800/velocity

    return sonic_transit


def density_si(bulk_density):

    """Convert the density to International System (I.S.).

    Parameters
    ----------
    bulk_density : int, float, array_like
        bulk_density values in standard g/cm3 log type.

    Returns
    -------
    density : int, float, array_like
        density in International System (Kg/m3).
    """

    density = bulk_density*1000.0

    return density


def density_sl(density):

    """Convert the density time to standard log system (g/cm3) .

    Parameters
    ----------
    density : int, float, array_like
        desnsity values in the International System (m/s).

    Returns
    -------
    sonic_transit : int, float, array_like
        sonic_transit in International System (us/ft).
    """

    bulk_density = density*.001

    return bulk_density


def neutron_porosity_fraction (neutron_porosity_percentage):

    """Convert the neutron porosity in standard percentage to neutron porosity in fraction .

    Parameters
    ----------
    neutron_porosity_percentage : int, float, array_like
        neutron porosity values in percentage.

    Returns
    -------
    neutron_fraction : int, float, array_like
        neutron porosity in fraction.
    """

    neutron_fraction = neutron_porosity_percentage/100.0

    return neutron_fraction


def neutron_porosity_percentage (neutron_porosity_fraction):

    """Convert the neutron porosity in fraction to neutron porosity in percentage.

    Parameters
    ----------
    neutron_porosity_fraction : int, float, array_like
        neutron porosity values in the fraction.

    Returns
    -------
    neutron_fraction : int, float, array_like
        neutron porosity values in percentage.
    """

    neutron_percentage = neutron_porosity_fraction*100.0

    return neutron_percentage
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    