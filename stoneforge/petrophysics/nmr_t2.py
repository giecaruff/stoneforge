import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def nmr_t2_ksdr_permeability(phi: npt.ArrayLike, t2_times: npt.NDArray, t2_dist_amps: npt.ArrayLike, a: float, b: float = 4.0, c: float = 2.0) -> np.ndarray:
    """Estimate permeability using k-SDR equation [1]. 
    It calculates the T2 log-mean (T2lm) of each given T2 distribution, 
    and calculates the equation -> k_sdr = a * (phi^b) * (T2lm^c).

    Parameters
    ----------
    phi : 1D array_like
        Porosity log.
    t2_times : 1D array_like
        The T2 time values of the T2 distributions. 
    t2_dist_amps : 1D or 2D array_like
        T2 distributions log.
    a : float
        k-SDR Equation's premultiplier factor, dependent on the formation's lithology. Optimized with core data.
    b : float
        Expoent factor for porosity, dependent on the formation's lithology. Optimized with core data, and defaults to 4.0.
    c : float
        Expoent factor for T2 log-mean, dependent on the formation's lithology. Optimized with core data, and defaults to 2.0.

    Returns
    -------
    k_sdr : 1D array_like
        Estimated Permeability Log.

    References
    ----------      
    .. [1] Kenyon, W. E., Day, P. I., Straley, C., Willemsen, J. F., 1988, A three-part study of NMR longitudinal relaxation
properties of water saturated sandstones: SPE Formation Evaluation, vol. 3, p. 622-636.

    """
    
    t2_log_means = 10.0 ** (np.sum(t2_dist_amps * np.log10(t2_times), axis=-1) / np.sum(t2_dist_amps, axis=-1))
    ksdr = a * np.power(phi, b) * np.power(t2_log_means, c)
    return ksdr


def nmr_t2_calculate_bvi_ffi(t2_times: npt.NDArray, t2_dist_amps: npt.NDArray, t2_cutoff : float, force_interpolation : bool = False):
    """Calculate Bound Volume Index (BVI) and Free Fluid Index (FFI) of T2 Distributions by applying a T2 cutoff.

    Parameters
    ----------
    t2_times : 1D array_like
        The T2 time values of the T2 distributions. 
    t2_dist_amps : 1D or 2D array_like
        T2 distributions log.
    t2_cutoff : float
        T2 cutoff. Must be in the same unit of "t2_times".
    force_interpolation : bool
        Use interpolation to improve the result (not fully implemented yet).

    Returns
    -------
    bvi : float or 1D array_like
        Calculated Bound Volume Index(es).
    ffi : float or 1D array_like
        Calculated Free Fluid Index(es).

    References
    ----------      
    .. [1] Kenyon, W. E., Day, P. I., Straley, C., Willemsen, J. F., 1988, A three-part study of NMR longitudinal relaxation
properties of water saturated sandstones: SPE Formation Evaluation, vol. 3, p. 622-636.

    """

    norm_t2_dist_amps = t2_dist_amps / t2_dist_amps.sum(axis=-1)
    if len(t2_dist_amps.shape) == 2:
        num_dists = t2_dist_amps.shape[0]
        bvi = np.empty(num_dists, dtype=t2_dist_amps.dtype)
        ffi = np.empty(num_dists, dtype=t2_dist_amps.dtype)
    else:
        num_dists = 1
    
    if False and (force_interpolation or t2_dist_amps.shape[-1] < 256): # needs more testing before implementation
        t2_dist_log10_times = np.log10(t2_times)
        t2_dist_interp_log10_times = np.linspace(t2_dist_log10_times.min(), t2_dist_log10_times.max(), 256)
        bvi_indexes = t2_dist_interp_log10_times <= np.log10(t2_swi_cutoff)
        if num_dists > 1:
            for si in range(num_dists):
                t2_dist_interp_curve = interp1d(t2_dist_log10_times, t2_dist_amps[si])(t2_dist_interp_log10_times)
                bvi[si] = np.sum((t2_dist_interp_curve / t2_dist_interp_curve.sum())[bvi_indexes])
        else:
            t2_dist_interp_curve = interp1d(t2_dist_log10_times, t2_dist_amps[si])(t2_dist_interp_log10_times)
            bvi = np.sum((t2_dist_interp_curve / t2_dist_interp_curve.sum())[bvi_indexes])
    else:
        bvi_indexes = t2_times <= t2_cutoff
        if num_dists > 1:
            for si in range(num_dists): 
                bvi[si] = np.sum(norm_t2_dist_amps[bvi_indexes])
        else:
            bvi = np.sum(norm_t2_dist_amps[bvi_indexes])
    ffi = 1.0 - bvi
    return bvi, ffi


def nmr_t2_calculate_k_timur_coates(phi : npt.ArrayLike, bvi : npt.ArrayLike, ffi : npt.ArrayLike, ktc_coeff : float):
    """Estimate permeability using the Timur-Coates equation [1].
    Calculates k_timur_coates = ((FFI/BVI) * (phi/ktc_coeff)²)²

    Parameters
    ----------
    phi : float or 1D array_like
        Porosity log.
    bvi : float or 1D array_like
        Bound Volume Index log.
    ffi : float or 1D array_like
        Free Fluid Index log.
    ktc_coeff : float
        Coefficient dependent on the formation's lithology. Optimized with core data.

    Returns
    -------
    k_timur_coates : float or 1D array_like
        Estimated Permeability Log.

    """

    k_timur_coates = (((phi/ktc_coeff)**2.0) * (ffi/bvi))**2.0  # k = ((FFI/BVI) * (phi/c)²)²
    return k_timur_coates


def nmr_t2_optimize_ksdr_coefficients(phi: npt.ArrayLike, t2_times: npt.NDArray, t2_dist_amps: npt.ArrayLike, kabs: npt.ArrayLike):
    """Find optimized values for the lithological coefficients of the k-SDR equation.
    Solves a, b, c by minimizing the equation -> k_sdr = a * (phi^b) * (T2lm^c).

    Parameters
    ----------
    phi : 1D array_like
        Porosity data.
    t2_times : 1D array_like
        The T2 time values of the T2 distributions. 
    t2_dist_amps : 1D or 2D ndarray
        T2 distributions log.
    kabs : 1D array_like
        Absolute Permeability data to fit.

    Returns
    -------
    a : float
        k-SDR Equation's premultiplier factor, dependent on the formation's lithology. Optimized with core data.
    b : float
        Expoent factor for porosity, dependent on the formation's lithology. Optimized with core data, and defaults to 4.0.
    c : float
        Expoent factor for T2 log-mean, dependent on the formation's lithology. Optimized with core data, and defaults to 2.0.
    R² : float
        Expoent factor for T2 log-mean, dependent on the formation's lithology. Optimized with core data, and defaults to 2.0.
    RMSEP : float
        Expoent factor for T2 log-mean, dependent on the formation's lithology. Optimized with core data, and defaults to 2.0.
    """

    guesses = np.array([0.01, 4.0, 2.0], dtype=np.float64)
    log_phi, log_kabs = np.log10(phi), np.log10(kabs)
    log_t2lms = np.sum(t2_dist_amps * np.log10(t2_times), axis=-1) / np.sum(t2_dist_amps, axis=-1)

    def __kck_min_sqr_func(x):
        lkd = log_kabs - (x[0] + (x[1] * log_phi) + (x[2] * log_t2lms))
        return np.sum(lkd * lkd)

    def __kck_min_sqr_jac(x):
        lkd = log_kabs - (x[0] + (x[1] * log_phi) + (x[2] * log_t2lms))
        return np.array([-2.0 * np.sum(lkd * jm) for jm in (1.0, log_phi, log_t2lms)])

    a, b, c = minimize(__kck_min_sqr_func, guesses, method='SLSQP', jac=__kck_min_sqr_jac, tol=1e-6).get('x')
    log_k_estimated = a + (b * log_phi) + (c * log_t2lms)
    difv = log_k_estimated - log_kabs
    dev_abs_v, dev_prd_v = log_kabs - np.mean(log_kabs), log_k_estimated - np.mean(log_k_estimated)
    r = np.sum(dev_abs_v * dev_prd_v) / np.sqrt(np.sum(dev_abs_v * dev_abs_v) * np.sum(dev_prd_v * dev_prd_v))
    rmsep = np.sqrt(np.sum(difv * difv) / difv.size)
    return a, b, c, r*r, rmsep