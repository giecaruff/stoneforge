import numpy as np
from typing import Annotated
import scipy
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gamma_calc(
      data: Annotated[np.array, "1D dataset"],
      depth: Annotated[np.array, "1D Depth data"],
      step: Annotated[int, "Step between samples"]=100):
      """Smooth the values of 1D data (:footcite:t:`isaaks1989,isaaks2013,dvorkin2014`).

      Parameters
      ----------
      data : array_like
            1D data samples From where the calculation will be made.
            
      depth : array_like
            1D array containing the depth of each sample in the dataset.
            
      step : int, optional
            Step between samples, by default 100. This is the number of samples that will be skipped between each calculation of the correlation function.

      Returns
      -------
      gamma : array_like
            1D array containing the calculated gamma values for each step.
      
      """
      depth_intervals = []
      gamma = []
      for st in range(step):
            
            _h = [] # head
            _t = [] # tail
            gamma_value = []

            for i in range(len(data)-st):
                  _t.append(data[i+st])
                  _h.append(data[i])
                  gamma_value.append( ((data[i+st] - data[i])**2) )

            depth_intervals.append(depth[0+st] - depth[0])
            gamma.append(sum(gamma_value)/(2*len(gamma_value)))

      return np.array(gamma),np.array(depth_intervals)


class variogram_model:
      """_summary_
      A class to calculate and visualize the variogram of a dataset, using the exponential model. It can also normalize the data and calculate the variogram for normalized data.
      It provides methods to graph the variogram, calculate the variogram for a given depth, and normalize and denormalize the data (:footcite:t:`isaaks1989,isaaks2013,dvorkin2014`).
      """

      def __init__(
            self: Annotated["variogram_model", "Variogram model class"],
            dif: Annotated[np.array, "1D dataset"],
            depth: Annotated[np.array, "1D Depth data"],
            step: Annotated[int, "Step between samples"]=100):
            """Initialize the variogram model with the given data, depth and step.
            
            Parameters
            ----------
            dif : array_like
                  1D data samples From where the calculation will be made.
                  
            depth : array_like
                  1D array containing the depth of each sample in the dataset.
                  
            step : int, optional
                  Step between samples, by default 100. This is the number of samples that will be skipped between each calculation of the correlation function.
            
            """
            self.dif = dif
            self.depth = depth
            self.step = step
            self.gm,self.dt = gamma_calc(dif,depth,step)

            self.min_dif = np.min(self.dif)
            self.max_dif = np.max(self.dif)
            self.dif_norm = (self.dif - self.min_dif) / (self.max_dif - self.min_dif)

      def graph(
            self: Annotated["variogram_model", "Variogram model class"],
            correlation_length: Annotated[float, "Variogram range"],
            sill: Annotated[bool, "considers the existence of sill"] = False,
            nugget: Annotated[float, "Variogram sill"] = 0.0):
            """Graph the variogram of the dataset using the exponential model.
            
            Parameters
            ----------
            dif : array_like
                  1D data samples From where the calculation will be made.
                  
            depth : array_like
                  1D array containing the depth of each sample in the dataset.
                  
            step : int, optional
                  Step between samples, by default 100. This is the number of samples that will be skipped between each calculation of the correlation function.
            
            """
            self.correlation_length = correlation_length
            if sill:
                  self.sill = sill
            else:
                  self.sill = np.var(self.dif)
                  self.nugget = nugget
                  self.var = exponential_variogram_model(distance = self.dt,correlation_length = correlation_length,sill = self.sill)
            
            plt.plot(self.dt,self.gm,'r.')
            plt.plot(self.dt,self.var,'b--')
            plt.xlabel('Depth interval - h')
            plt.ylabel('Variogram - $\gamma(h)$')
            plt.grid()
            plt.show()
        
      def norm_graph(
            self: Annotated["variogram_model", "Variogram model class"],
            correlation_length: Annotated[float, "Variogram range"],
            sill: Annotated[bool, "considers the existence of sill"] = False,
            nugget: Annotated[float, "Variogram sill"] = 0.0):
            """Graph the normalized variogram of the dataset using the exponential model."""
            if sill:
                  self.n_sill = sill
            else:
                  self.n_sill = np.var(self.dif_norm)
            self.gm,self.dt = gamma_calc(self.dif_norm,self.depth,self.step)
            self.var =  exponential_variogram_model(distance = self.dt,correlation_length = correlation_length,sill = self.n_sill)
            self.n_correlation_length = correlation_length
            self.n_nugget = nugget

            plt.plot(self.dt,self.gm,'r.')
            plt.plot(self.dt,self.var,'b--')
            plt.xlabel('Depth interval - h')
            plt.ylabel('Variogram - $\gamma(h)$')
            plt.grid()
            plt.show()

      def variography(
            self: Annotated["variogram_model", "Variogram model class"],
            depth: Annotated[np.array, "1D Depth data"] = False):
            """Calculate the variogram of the dataset using the exponential model."""
            if depth is not False:
                  l_depth = depth
            else:
                  l_depth = self.depth
            return self._variogram(l_depth, self.correlation_length, self.sill, self.nugget)

      def norm_variography(
            self: Annotated["variogram_model", "Variogram model class"],
            depth: Annotated[np.array, "1D Depth data"] = False):
            """Calculate the normalized variogram of the dataset using the exponential model."""
            if depth is not False:
                  l_depth = depth
            else:
                  l_depth = self.depth
            return self._variogram(l_depth, self.n_correlation_length, self.n_sill, self.n_nugget)
    
      def normalization(
            self: Annotated["variogram_model", "Variogram model class"],
            data: Annotated[np.array, "1D data"]):
            """Normalize the data using the min-max normalization method."""
            return (data - self.min_dif) / (self.max_dif - self.min_dif)
    
      def denormalization(
            self: Annotated["variogram_model", "Variogram model class"],
            data: Annotated[np.array, "1D data"]):
            """Denormalize the data using the min-max normalization method."""
            return data * (self.max_dif - self.min_dif) + self.min_dif
    
      def _variogram(
            self: Annotated["variogram_model", "Variogram model class"],
            depths: Annotated[np.array, "1D Depth data"],
            a: Annotated[float, "Variogram range"],
            C1: Annotated[float, "Variogram sill"],
            C0: Annotated[float, "Variogram nugget"] = 0.0):

            X, Y = np.meshgrid(depths, depths)
            h = np.abs(X - Y)

            return np.where(h == 0, C0 + C1, C1 * np.exp((-3 * h) / a))

def experimental_correlation(
      data: Annotated[np.array, "1D dataset"])-> np.array:
      """Determines the 1D experimental correlation function for a dataset by calculating the Pearson correlation coefficient for each possible separation of samples (:footcite:t:`dvorkin2014`).

      Parameters
      ----------
      data : array_like
            1D dataset for which the experimental correlation function must be calculated.

      Returns
      -------
      rho : array_like
            1D experimental correlation function of the data under examination.
            
      """
      rho = np.zeros(len(data))
      for i in range(len(data) - 1):
            slc = (slice(0, len(data) - i, None), slice(i, len(data), None))
            rho[i] = pearsonr(data[slc[0]], data[slc[1]])[0]
      return(rho)


def experimental_variogram(
      data: Annotated[np.array, "1D dataset"],
      rho: Annotated[np.array, "1D correlation function data"]) -> np.array:
      """Determines the 1D experimental variogram of a dataset (:footcite:t:`dvorkin2014`).

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

      """
      return (np.std(data)**2) * (1 - rho)


def exponential_variogram_model(
      distance: Annotated[np.array, "1D array of distances"],
      correlation_length: Annotated[float, "Variogram range"],
      sill: Annotated[float, "Variogram sill"],
      nugget: Annotated[float, "Variogram sill"] = 0) -> np.array:
      """Builds a variogram following the exponential model, using the correlation length, sill and nugget given (:footcite:t:`journel1978scipy-models,cressie1993scipy-models,chiles1999scipy-models,grana2021`).

      Parameters
      ----------
      distance : array_like
            1D array containing all the possible distances between a pair of points in the dataset. 

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

      """   
      return nugget + sill * (1. - np.exp(-(3*distance/correlation_length)))


def gaussian_variogram_model(
      distance: Annotated[np.array, "1D array of distances"],
      correlation_length: Annotated[float, "Maximum length where correlation still occurs"],
      sill: Annotated[float, "Variogram sill"],
      nugget: Annotated[float, "Variogram sill"] = 0) -> np.array:
      """Builds a variogram following the gaussian model, using the correlation length, sill and nugget given (:footcite:t:`journel1978scipy-models,chiles1999scipy-models,grana2021`).

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

      """   
      return nugget + sill * (1. - np.exp(- 3*(distance ** 2 / correlation_length ** 2)))


def spherical_variogram_model(
      distance: Annotated[np.array, "1D array of distances"],
      correlation_length: Annotated[float, "Maximum length where correlation still occurs"],
      sill: Annotated[float, "Variogram sill"],
      nugget: Annotated[float, "Variogram sill"] = 0) -> np.array:
      """Builds a variogram following the spherical model, using the correlation length, sill and nugget given (:footcite:t:`burgess1980scipy-models,grana2021`).

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


def analytical_variogram(
      distance: Annotated[np.array, "1D array of distances"],
      gama: Annotated[np.array, "1D experimental variogram"],
      initial_guess: Annotated[np.array, "Initial guess"]) -> np.array:
      """Fits the choosen analytical variogram function (model) to the experimental one :footcite:t:`vugrin2007scipy-curve-fit`, if no model is choosen, determines the best model to fit, comparing the Gaussian,  Exponential and Spherical models :footcite:t:`grana2021,anacarolina2023`.

      Parameters
      ----------
      distance : array_like
            1D array containing all the possible distances between a pair of points in the dataset. 
  
      gama : array_like
            1D experimental variogram of the data under examination.

      model : str, optional
            Analytical variogram model to be fitted.  Should be one of:
            
            - "exponential": fits the exponential model
            - "gaussian": fits the gaussian model
            - "spherical": fits the spherical model
            - "best-fit": fits the three models above and verifies which one produces the smallest error.
            
      If not given, default method is "best-fit".

      initial_guess : array_like
            Initial guess for the parameters of the variogram model. It should be a list containing the initial values for the correlation length, sill and nugget.

      Returns
      -------
      modeled_variogram : array_like
            The variogram model that has been choosen, or the variogram model that fits the best the experimental one.
      coeficients : array_like
            The range, sill and nugget optimal values for the modeled variogram.

      """ 
      model_data = []

      xi = distance
      coeficients, cov = curve_fit(spherical_variogram_model, distance, gama, initial_guess)                               
      yi = list(map(lambda distance: spherical_variogram_model(distance, *coeficients), xi))        
      spherical_data = ["spherical", yi, coeficients, False]

      xig = distance
      coeficientsg, covg = curve_fit(gaussian_variogram_model, distance, gama, initial_guess)
      yig = list(map(lambda distance: gaussian_variogram_model(distance, *coeficientsg), xig))
      gaussian_data = ["gaussian", yig, coeficientsg, False]

      xie = distance
      coeficientse, cove = curve_fit(exponential_variogram_model, distance, gama, initial_guess)
      yie = list(map(lambda distance: exponential_variogram_model(distance, *coeficientse), xie))
      exponential_data = ["exponential", yie, coeficientse, False]

      ranges = np.array([coeficients[0],coeficientsg[0],coeficientse[0]])
      structured_field = distance <= np.max(ranges)

      difference_sph = np.zeros(len(gama))
      difference_gauss = np.zeros(len(gama))
      difference_exp = np.zeros(len(gama))

      i = 0
      while (structured_field[i] == True and i < (len(structured_field)-1)):
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
            spherical_data[3] = True
      if best == 1:
            gaussian_data[3] = True
      if best == 2:
            exponential_data[3] = True

      model_data.append(spherical_data)
      model_data.append(gaussian_data)
      model_data.append(exponential_data)

      return model_data


def modeled_correlation(
      gama: Annotated[np.array, "1D experimental variogram"],
      var: Annotated[float, "Variance"])-> np.array:
      """Determines the 1D modeled correlation function from a variogram model (:footcite:t:`dvorkin2014`).

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

      """
      return 1 - gama/var


def cov_matrix(
      rho: Annotated[np.array, "1D correlation function data"],
      var: Annotated[float, "Variance"])-> np.array:
      """Determines the 1D spatial symmetrical covariance matrix from a modeled correlation function (:footcite:t:`dvorkin2014`).

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

      """
      return scipy.linalg.toeplitz(var*rho)


def MCS_spacial_correlation(
      n: Annotated[int, "Number of simulations"],
      smooth_data: Annotated[np.array, "Smoothed version of the data"],
      cov: Annotated[np.array, "2D Spatial symmetrical covariance matrix"]) -> np.array:
      """Determines n Monte Carlo Simulations (MCS) with spatial correlation for a given (:footcite:t:`dvorkin2014`). 

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
            n Monte Carlo simulations with spatial correlation for a given property, each line of this matrix represents a different simulation.

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


def p(
      n: Annotated[int, "Number of simulations"],
      data1: Annotated[np.array, "Dataset porperty, related to data1"],
      data2: Annotated[np.array, "Dataset porperty, related to data2"],
      smooth_data1: Annotated[np.array, "General trend of data1"],
      smooth_data2: Annotated[np.array, "General trend of data1"],
      cov: Annotated[np.array, "2D Spatial symmetrical covariance matrix"])-> np.array:
      """Determines n Monte Carlo Simulations (MCS) using data1 and data2 as correlated variables (:footcite:t:`dvorkin2014`).

      Parameters
      ----------
      n : integer
            Number of simulations to be performed.
      data1 : array_like
            A dataset that represents a given porperty, related to data1.
      data2 : array_like
            A dataset that represents a given porperty, related to data2.
      smooth_data1 : array_like
            A smoothed version of the data1, or its general trend.
      smooth_data2 : array_like
            A smoothed version of the data2, or its general trend.
      cov : array_like
            Spatial symmetrical covariance matrix representing both data1 and data2.

      Returns
      -------
      simulations : array_like
            n Monte Carlo Simulations with correlated variables for data1 and n Monte
            Carlo Simulations with correlated variables for data2, in this order.

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