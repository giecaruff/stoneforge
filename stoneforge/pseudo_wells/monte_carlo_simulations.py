import numpy as np
import numpy.typing as npt
import scipy
from scipy import stats
from scipy.stats import pearsonr
from scipy.optimize import curve_fit


def experimental_correlation(data: npt.ArrayLike)-> np.ndarray:
  """
  Determines the 1D experimental correlation function [1]_ for a dataset by calculating 
  the Pearson correlation coefficient for each possible separation of samples.

  Parameters
  ----------
  data : array_like
        1D dataset for which the experimental correlation function must be calculated.

  Returns
  -------
  rho : array_like
        1D experimental correlation function of the data under examination.
  
  References
  ----------
  .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
  properties. [S.l.]: Cambridge University Press, 2014.

  """
  rho = np.zeros(len(data))
  for i in range(len(data) - 1):
        slc = (slice(0, len(data) - i, None), slice(i, len(data), None))
        rho[i] = pearsonr(data[slc[0]], data[slc[1]])[0]
  return(rho)


def experimental_variogram(data: npt.ArrayLike, rho: npt.ArrayLike)-> np.ndarray:
  """
  Determines the 1D experimental variogram [1]_ of a dataset.

  Parameters
  ----------
  data : array_like
        1D dataset for which the experimental variogram must be calculated.
  
  rho : array_like
        1D experimental correlation function of the data under examination.

  Returns
  -------
  gama : array_like
        1D experimental variogram of the data under examination.
  
  References
  ----------
  .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
  properties. [S.l.]: Cambridge University Press, 2014.

  """
  gama = (np.std(data)**2) * (1 - rho)
  return(gama)


def exponential_variogram_model(distance: npt.ArrayLike, correlation_length: float, sill: float, nugget: float=0)-> np.ndarray:
  """
  Builds a variogram following the exponential model, using the correlation length, sill and nugget given [1]_.

  Parameters
  ----------
  distance : array_like
        1D array containing all the possible distances between a pair of points
        in the dataset. 

  correlation_length : float
        The range of the variogram, or the distance where it loses the correlation 

  sill : float
        The maximum value of the variogram, it is equivalent to the variance of the data 

  nugget : float
        The nugget effect, y value where the variogam begins

  Returns
  -------
  rho : array_like
        The variogram that follows the exponential model and has the given correlation length, sill and nugget

  References
  ----------
  .. [1] GRANA, Dario; MUKERJI, Tapan; DOYEN, Philippe. Seismic Reservoir Modeling: Theory,
  Examples and Algorithms. India: Wiley Blackwell, 2021.

  """   
  rho = nugget + sill * (1. - np.exp(-(3*distance/correlation_length)))
  return(rho)

def gaussian_variogram_model(distance: npt.ArrayLike, correlation_length: float, sill: float, nugget: float=0)-> np.ndarray:
  """
  Builds a variogram following the gaussian model, using the correlation length, sill and nugget given [1]_.

  Parameters
  ----------
  distance : array_like
        1D array containing all the possible distances between a pair of points
        in the dataset. 

  correlation_length : float
        The range of the variogram, or the distance where it loses the correlation 

  sill : float
        The maximum value of the variogram, it is equivalent to the variance of the data 

  nugget : float
        The nugget effect, y value where the variogam begins

  Returns
  -------
  rho : array_like
        The variogram that follows the gaussian model and has the given correlation length, sill and nugget

  References
  ----------
  .. [1] GRANA, Dario; MUKERJI, Tapan; DOYEN, Philippe. Seismic Reservoir Modeling: Theory,
  Examples and Algorithms. India: Wiley Blackwell, 2021.

  """   
  rho = nugget + sill * (1. - np.exp(- 3*(distance ** 2 / correlation_length ** 2)))
  return(rho)

def spherical_variogram_model(distance: npt.ArrayLike, correlation_length: float, sill: float, nugget: float=0)-> np.ndarray:
  """
  Builds a variogram following the spherical model, using the correlation length, sill and nugget given [1]_.

  Parameters
  ----------
  distance : array_like
        1D array containing all the possible distances between a pair of points
        in the dataset. 

  correlation_length : float
        The range of the variogram, or the distance where it loses the correlation 

  sill : float
        The maximum value of the variogram, it is equivalent to the variance of the data 

  nugget : float
        The nugget effect, y value where the variogam begins

  Returns
  -------
  rho : array_like
        The variogram that follows the spherical model and has the given correlation length, sill and nugget

  References
  ----------
  .. [1] GRANA, Dario; MUKERJI, Tapan; DOYEN, Philippe. Seismic Reservoir Modeling: Theory,
  Examples and Algorithms. India: Wiley Blackwell, 2021.

  """   
  rho = []
  try:
        for i in range(len(distance)):
              if distance[i] <= correlation_length:
                    rho.append(nugget + sill * ((1.5 * (distance[i] / correlation_length)) - (0.5 * ((distance[i] / correlation_length) ** 3.0))))
              else:
                    rho.append(nugget + sill)
        return(rho)
  
  except TypeError:
        if distance <= correlation_length:
              rho = nugget + sill * ((1.5 * (distance / correlation_length)) - (0.5 * ((distance / correlation_length) ** 3.0)))
        else:
              rho = nugget + sill
        return(rho)


def analytical_variogram(distance: npt.ArrayLike, gama: npt.ArrayLike, model:str="best-fit")-> np.ndarray:
  """
  Fits the choosen analytical variogram function (model) to the experimental one [1]_, 
  if no model is choosen, determines the best model to fit, comparing the Gaussian, 
  Exponential and Spherical models [2]_.

  Parameters
  ----------
  distance : array_like
        1D array containing all the possible distances between a pair of points
        in the dataset. 
  
  gama : array_like
        1D experimental variogram of the data under examination.

  model : str, optional
        Analytical variogram model to be fitted.  Should be one of:
            - "exponential": fits the exponential model
            - "gaussian": fits the gaussian model
            - "spherical": fits the spherical model
            - "best-fit": fits the three models above and verifies which one produces
              the smallest error.
        If not given, default method is "best-fit".

  Returns
  -------
  modeled_variogram : array_like
        The variogram model that has been choosen, or the variogram model that fits the
        best the experimental one.
  range : float
        The range of the modeled variogram.

  References
  ----------
  .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
  .. [2] GRANA, Dario; MUKERJI, Tapan; DOYEN, Philippe. Seismic Reservoir Modeling: Theory,
  Examples and Algorithms. India: Wiley Blackwell, 2021.

  """ 
  if model == "spherical":
    xi = distance
    coeficients, cov = curve_fit(spherical_variogram_model, distance, gama)                               
    yi = list(map(lambda distance: spherical_variogram_model(distance, *coeficients), xi))
    return(yi,coeficients[0])

  elif model == "gaussian":
    xig = distance
    coeficientsg, covg = curve_fit(gaussian_variogram_model, distance, gama)
    yig = list(map(lambda distance: gaussian_variogram_model(distance, *coeficientsg), xig))
    return(yig,coeficientsg[0])

  elif model == "exponential":
    xie = distance
    coeficientse, cove = curve_fit(exponential_variogram_model, distance, gama)
    yie = list(map(lambda distance: exponential_variogram_model(distance, *coeficientse), xie))
    return(yie,coeficientse[0])

  elif model == "best-fit":
    xi = distance
    coeficients, cov = curve_fit(spherical_variogram_model, distance, gama)                               
    yi = list(map(lambda distance: spherical_variogram_model(distance, *coeficients), xi))        

    xig = distance
    coeficientsg, covg = curve_fit(gaussian_variogram_model, distance, gama)
    yig = list(map(lambda distance: gaussian_variogram_model(distance, *coeficientsg), xig))

    xie = distance
    coeficientse, cove = curve_fit(exponential_variogram_model, distance, gama)
    yie = list(map(lambda distance: exponential_variogram_model(distance, *coeficientse), xie))

    ranges = np.array([coeficients[0],coeficientsg[0],coeficientse[0]])
    structured_field = distance <= np.max(ranges)

    difference_sph = np.zeros(len(gama))
    difference_gauss = np.zeros(len(gama))
    difference_exp = np.zeros(len(gama))

    i = 0
    while (structured_field[i] == True):
      i += 1
      difference_sph[i] = gama[i] - yi[i]
      difference_gauss[i] = gama[i] - yig[i]
      difference_exp[i] = gama[i] - yie[i]

    rmse_sph = ((np.sum(difference_sph**2))/i)**0.5
    rmse_gaus = ((np.sum(difference_gauss**2))/i)**0.5
    rmse_exp = ((np.sum(difference_exp**2))/i)**0.5

    RMSE = np.array((rmse_sph, rmse_gaus, rmse_exp))
    best = list(RMSE).index(np.min(RMSE))

    if best == 0:
      return(yi,coeficients[0])
    if best == 1:
      return(yig,coeficientsg[0])
    if best == 2:
      return(yie,coeficientse[0])
  else:
    raise TypeError("model must be: exponential, gaussian, spherical or best-fit")


def modeled_correlation(gama: npt.ArrayLike, var: float)-> np.ndarray:
  """
  Determines the 1D modeled correlation function [1]_ from a variogram model [2]_.

  Parameters
  ----------
  gama : array_like
        1D analytical variogram model.
  var  : float
         variance (the square of the standard deviation) of the dataset.

  Returns
  -------
  rho : array_like
        1D modeled correlation function of the data under examination.
  
  References
  ----------
  .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
  properties. [S.l.]: Cambridge University Press, 2014.
  .. [2] https://mmaelicke.github.io/scikit-gstat/reference/models.html
  """
  rho = 1 - gama/var
  return(rho)


def cov_matrix(rho: npt.ArrayLike, var: float)-> np.ndarray:
  """
  Determines the 1D spatial symmetrical covariance matrix [1]_ from a modeled 
  correlation function.

  Parameters
  ----------
  rho : array_like
        1D modeled correlation function of the data under examination.
  var  : float
         Variance (the square of the standard deviation) of the dataset.

  Returns
  -------
  cov : array_like
       Spatial symmetrical covariance matrix of the data.
  
  References
  ----------
  .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
  properties. [S.l.]: Cambridge University Press, 2014.

  """
  cov = scipy.linalg.toeplitz(var*modelo)
  return(cov)


def MCS_spacial_correlation(n: int, smooth_data: npt.ArrayLike, cov: npt.ArrayLike)-> np.ndarray:
  """
  Determines n Monte Carlo Simulations (MCS) with spatial correlation [1]_ for a given 
  dataset. 

  Parameters
  ----------
  n : integer
        Number of simulations to be performed.
  smooth_data  : array_like
        a smoothed version of the data under examination, or its general trend.
  cov : array_like
       Spatial symmetrical covariance matrix of the data.

  Returns
  -------
  simulations : array_like
       n Monte Carlo simulations with spatial correlation for a given property, 
       each line of this matrix represents a different simulation.
  
  References
  ----------
  .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
  properties. [S.l.]: Cambridge University Press, 2014.

  """
  simulations = np.zeros((n, len(smooth_data)))
  w = np.zeros((n, len(smooth_data)))
  v = np.zeros((n, len(smooth_data)))
  R = np.linalg.cholesky(cov)

  for i in range(n):
      u = np.random.normal(loc=0, scale=1, size=len(smooth_data)) 
      simulations[i, :] = u
      w[i, :] = np.dot(R, simulations[i, :])                                    
      v[i, :] = smooth_data + w[i, :]   

  return(simulations)


def MCS_correlated_variables(n: int, data1: npt.ArrayLike, data2: npt.ArrayLike,
                             smooth_data1: npt.ArrayLike, smooth_data2: npt.ArrayLike,
                             cov: npt.ArrayLike)-> np.ndarray:
  """
  Determines n Monte Carlo Simulations (MCS) using data1 and data2 as correlated 
  variables [1]_.

  Parameters
  ----------
  n : integer
        Number of simulations to be performed.
  data1  : array_like
        A dataset that represents a given porperty, related to data2.
  data2  : array_like
        A dataset that represents a given porperty, related to data1.
  smooth_data1  : array_like
        A smoothed version of the data1, or its general trend.
  smooth_data2  : array_like
        A smoothed version of the data2, or its general trend.
  cov : array_like
       Spatial symmetrical covariance matrix representing both data1 and data2.

  Returns
  -------
  simulations : array_like
       n Monte Carlo Simulations with correlated variables for data1 and n Monte
       Carlo Simulations with correlated variables for data2, in this order.
  
  References
  ----------
  .. [1] Dvorkin, J.; Gutierrez, M. A.; Grana, D. Seismic reflections of rock
  properties. [S.l.]: Cambridge University Press, 2014.

  """
  S = np.cov(data1, data2)
  K = np.kron(S, cov)
  R = np.linalg.cholesky(K)

  simulations = np.zeros((n, 2*len(data1)))
  w = np.zeros((n, 2*len(data1)))
  v = np.zeros((n, 2*len(data1)))
  mm = np.concatenate((smooth_data1, smooth_data2))

  for i in range(n):
      u = np.random.normal(loc=0, scale=1, size=2*len(data1))                
      simulations[i, :] = u
      w[i, :] = np.dot(R, simulations[i, :])                                   
      v[i, :] = mm + w[i, :]      

  vv1 = np.zeros((n, len(data1)))
  vv2 = np.zeros((n, len(data1))) 
  for i in range(n):
      vv1[i, :] = v[i, 0:len(data1)]
      vv2[i, :] = v[i, len(data1):]
                        
  simulations = (vv1,vv2)  

  return(simulations)