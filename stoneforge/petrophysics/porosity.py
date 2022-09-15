import numpy as np
import numpy.typing as npt
import warnings


def density_porosity(rhob: npt.ArrayLike, rhom: float, rhof: float, depth:npt.ArrayLike=[]) -> np.ndarray:
    """Estimate the porosity from the bulk density log [1]_.

    Parameters
    ----------
    rhob : array_like
        Bulk density log.
    rhom : int, float
        Matrix density.
    rhof : int, float
        Density of the fluid saturating the rock (Usually 1.0 for water and 1.1 for saltwater mud).
    depth : array_like, optional
        depth array from the well, used to indicate the depths where there are unexpected values of porosity (negative or greater than 1)
       
    Returns
    -------
    phi : array_like
        Total porosity for the aimed interval using the bulk density.

    References
    ----------      
    .. [1] Schön, J. H. (2015). Physical properties of rocks: Fundamentals and 
    principles of petrophysics. Elsevier.

    """
    if rhom == rhof:
        warnings.warn(UserWarning("This will result in a division by zero"))
        return np.Inf

    else:
        phi = (rhom - rhob) / (rhom - rhof)

        if rhom < rhof or any(rhom <= rhob):
            warnings.warn(UserWarning("rhom must be greater than rhof and rhob, values that didn't respect that were replaced by nan"))
            phi = np.where(phi>0, phi, np.nan)
        
        if any(rhom - rhob > rhom - rhof):
            warnings.warn(UserWarning("phi must be a value between 0 and 1, values that didn't respect that were replaced by nan"))
            phi = np.where(phi<1, phi, np.nan)
            
        if len(depth) != 0:
          aux_boolean = np.isnan(phi)
          indexes = np.where(aux_boolean==True)
          lista_de_nans = []
          for i in range(len(indexes)):
            lista_de_nans = (depth[indexes[i]])
          if len(lista_de_nans) != 0:
            print( "These depths returned values for porosity that were not between 0 and 1: \n",lista_de_nans)
            return phi, lista_de_nans

        return phi
    

def neutron_porosity(nphi: npt.ArrayLike, vsh: npt.ArrayLike,
             nphi_sh: float, depth:npt.ArrayLike=[]) -> np.ndarray:
    """Estimate the effective porosity from the neutron log [1]_.

    Parameters
    ----------
    nphi : array_like
        neutron log.
    vsh : array_like
        Total volume of shale in the rock, chosen the most representative.
    phi_nsh : int, float
        Apparent porosity read in the shales on and under the layer under study and with the same values used in φN.
    depth : array_like, optional
        depth array from the well, used to indicate the depths where there are unexpected values of porosity (negative or greater than 1)
       
    Returns
    -------
    phin : array_like
        Effective porosity from the neutron log for the aimed interval.

    References
    ----------
    .. [1] Schön, J. H. (2015). Physical properties of rocks: Fundamentals and 
    principles of petrophysics. Elsevier.

    """
    phin = nphi - (vsh * nphi_sh)

    if any(nphi < (vsh * nphi_sh)):
        warnings.warn(UserWarning("phin must be a positive value, values that didn't respect that were replaced by nan"))
        phin = np.where(phin>0, phin, np.nan)
    
    if any(nphi - (vsh * nphi_sh) > 1):
        warnings.warn(UserWarning("phin must be a value between 0 and 1, values that didn't respect that were replaced by nan"))
        phin = np.where(phin<1, phin, np.nan)

    if len(depth) != 0:
      aux_boolean = np.isnan(phin)
      indexes = np.where(aux_boolean==True)
      lista_de_nans = []
      for i in range(len(indexes)):
        lista_de_nans = (depth[indexes[i]])
      if len(lista_de_nans) != 0:
        print( "These depths returned values for porosity that were not between 0 and 1: \n",lista_de_nans)
        return phin, lista_de_nans

    return phin


def neutron_density_porosity(phid: npt.ArrayLike, phin: npt.ArrayLike,
                squared: bool = False, depth:npt.ArrayLike=[]) -> np.ndarray:
    """Estimate the effective porosity by calculating the mean of Bulk Density porosity and Neutron porosity [1]_.

    Parameters
    ----------
    phid : array_like
        Effective porosity and shale free for the aimed interval using the bulk density.
    phin : array_like
        Effective porosity from the neutron log for the aimed interval.
    depth : array_like, optional
        depth array from the well, used to indicate the depths where there are unexpected values of porosity (negative or greater than 1)
    
    Returns
    -------
    phie : array_like
        Effective porosity from the Bulk Density porosity and Neutron porosity mean.

    References
    ----------
    TODO

    """
    if squared == False:
        phi = (phid + phin) / 2
        if any((phid + phin / 2) > 1):
            warnings.warn(UserWarning("phi must be a value between 0 and 1, values that didn't respect that were replaced by nan"))
            phi = np.where(phi>0, phi, np.nan)

    elif squared == True:
        phi = np.sqrt( (phid**2 + phin**2) / 2)
        if any((phid**2 + phin**2 / 2) > 1):
            warnings.warn(UserWarning("phi must be a value between 0 and 1, values that didn't respect that were replaced by nan"))
            phi = np.where(phi<1, phi, np.nan)

    if len(DEPTH) != 0:
      aux_boolean = np.isnan(phi)
      indexes = np.where(aux_boolean==True)
      lista_de_nans = []
      for i in range(len(indexes)):
        lista_de_nans = (DEPTH[indexes[i]])
      if len(lista_de_nans) != 0:
        print( "These depths returned values for porosity that were not between 0 and 1: \n",lista_de_nans)
        return phi, lista_de_nans

    return phi  


#TODO phit -> phie (clay volume correction)


def sonic_porosity(dt, dtma, dtf, depth:npt.ArrayLike=[]):
    """Estimate the Porosity from sonic using the Wyllie time-average equation [1]_.

    Parameters
    ----------
    dt : array_like
        Sonic log reading (acoustic transit time (μsec/ft))
    dtma : int, float
        Acoustic transit time of the matrix (μsec/ft)
    dtf : int, float
        Acoustic transit time of the fluids, usually water (μsec/ft)
    depth : array_like, optional
        depth array from the well, used to indicate the depths where there are unexpected values of porosity (negative or greater than 1)
                  
    Returns
    -------
    phidt : array_like
        Porosity from sonic.

    References
    ----------
    .. [1] M. R. J. Wyllie, A. R. Gregory, and L. W. Gardner, (1956), "ELASTIC WAVE VELOCITIES IN HETEROGENEOUS AND POROUS MEDIA," GEOPHYSICS 21: 41-70.

    """
    if dtf == dtma:
        warnings.warn(UserWarning("This will result in a division by zero"))
        return np.Inf

    else:
        phidt = (dt - dtma) / (dtf - dtma)

        if any(dt <= dtma) or dtf <= dtma:
            warnings.warn(UserWarning("dt and dtf must be greater than dtma, values that didn't respect that were replaced by nan"))
            phidt = np.where(phidt>0, phidt, np.nan)

        if any(dt - dtma > dtf - dtma):
            warnings.warn(UserWarning("phidt must be between 0 and 1, values that didn't respect that were replaced by nan"))
            phidt = np.where(phidt<1, phidt, np.nan)

        if len(DEPTH) != 0:
          aux_boolean = np.isnan(phidt)
          indexes = np.where(aux_boolean==True)
          lista_de_nans = []
          for i in range(len(indexes)):
            lista_de_nans = (DEPTH[indexes[i]])
          if len(lista_de_nans) != 0:
            print( "These depths returned values for porosity that were not between 0 and 1: \n",lista_de_nans)
            return phidt, lista_de_nans
    
    return phidt

def gaymard_porosity(phid, phin):
    """Estimate the effective porosity using Gaymard-Poupon [1]_ method.

    Parameters
    ----------
    phid : array_like
        Density porosity (porosity calculated using density log)
    phin : int, float
        Neutron porosity (porosity calculated using neutron log)

    Returns
    -------
    phie : array_like
        Effective porosity using Gaymard-Poupon method
    
    References
    ----------
    .. [1] Gaymard, R., and A. Poupon. "Response Of Neutron And Formation
    Density Logs In Hydrocarbon Bearing Formations." The Log Analyst 9 (1968).

    """
    phie = (0.5 * (phid*phid + phin*phin)) ** 0.5

    return phie


_porosity_methods = {
    "density": density_porosity,
    "neutron": neutron_porosity,
    "neutron-density": neutron_density_porosity,
    "sonic": sonic_porosity,
    "gaymard": gaymard_porosity
}


def porosity(method: str = "density", **kwargs):
    """Compute porosity from well logs.

    This is a façade for the methods:
        - density
        - neutron
        - neutron-density
        - sonic
        - gaymard

    Parameters
    ----------
    rhob : array_like
        Bulk density log. Required if `method` is "denisty".
    rhom : int, float
        Matrix density. Required if `method` is "denisty".
    rhof : int, float
        Density of the fluid saturating the rock (Usually 1.0 for water and 1.1 for saltwater mud). Required if `method` is "denisty".
    nphi : array_like
        Neutron log. Required if `method` is "neutron".
    vsh : array_like
        Total volume of shale in the rock, chosen the most representative. Required if `method` is "neutron".
    phi_nsh : int, float
        Apparent porosity read in the shales on and under the layer under study and with the same values used in φN. Required if `method` is "neutron".
    dt : array_like
        Sonic log reading (acoustic transit time (μsec/ft)). Required if `method` is "sonic".
    dtma : int, float
        Acoustic transit time of the matrix (μsec/ft). Required if `method` is "sonic".
    dtf : int, float
        Acoustic transit time of the fluids, usually water (μsec/ft). Required if `method` is "sonic".
    phid : array_like
        Density porosity (porosity calculated using density log). Required if `method` is "neutron-density" or "gaymard.
    phin : int, float
        Neutron porosity (porosity calculated using neutron log). Required if `method` is "neutron-density" or "gaymard.
    method : str, optional
        Name of the method to be used.  Should be one of
            - 'density'
            - 'neutron'
            - 'neutron-density'
            - 'sonic'
            - 'gaymard'
        If not given, default method is 'density'

    Returns
    -------
    phi : array_like
        Porosity log using the defined method.

    """
    options = {}

    required = []
    if method == "density":
        required = ["rhob", "rhom", "rhof"]
    elif method == "neutron":
        required = ["nphi", "vsh", "nphi_sh"]
    elif method == "neutron-density":
        required = ["phid", "phin"]
    elif method == "sonic":
        required = ["dt", "dtma", "dtf"]
    elif method == "gaymard":
        required = ["phid", "phin"]

    for arg in required:
        if arg not in kwargs:
            msg = f"Missing required argument for method '{method}': '{arg}'"
            raise TypeError(msg)
        options[arg] = kwargs[arg]

    fun = _porosity_methods[method]

    return fun(**options)
