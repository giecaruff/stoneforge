import numpy as np


def log_statstics(log,lito):

    sequence = lito[~np.isnan(lito)]
    states = list(set(sequence))

    p = len(lito)

    _stats = {}
    for j in states:
        _local = []
        for i in range(p):
            if lito[i] == j:
                _local.append(log[i])
        _stats[j] = [np.nanmean(_local),np.nanstd(_local)]

    return _stats


def synthetic_log(stats, lithology = False, seed=42):
    
    np.random.seed(seed)
    _S = []
    for i in range(len(lithology)):
        if np.isnan(lithology[i]):
            _S.append(np.nan)
        else:
            _l = lithology[i]
            _m = stats[_l][0]
            _s = stats[_l][1]
            _S.append(np.random.normal(_m, _s, 1)[0])

    return _S


def moving_average(curve, step=100):
    """Note: only works in odd step values, but it's very fast
    """
    # Convert input curve to numpy array
    curve_array = np.array(curve)
    step = step + step%2

    extended_curve = np.pad(curve_array, (step//2, step//2), mode='edge')

    rolled = np.lib.stride_tricks.sliding_window_view(extended_curve, window_shape=(step+1,))
    
    smooth_curve = np.mean(rolled, axis=1)

    return smooth_curve


def gamma_calc(dif_curve,depth,step=100):

    depth_intervals = []
    gamma = []
    for st in range(step): # 'step' from 0 to 300

    # ------------------------------------------ #
        
        _h = [] # head
        _t = [] # tail
        gamma_value = []
    
        for i in range(len(dif_curve)-st):
            _t.append(dif_curve[i+st])
            _h.append(dif_curve[i])
            gamma_value.append( ((dif_curve[i+st] - dif_curve[i])**2) )
    
        depth_intervals.append(depth[0+st] - depth[0])
        gamma.append(sum(gamma_value)/(2*len(gamma_value)))

    return np.array(gamma),np.array(depth_intervals)

def adjustment(dept,a,C1,C0 = 0,mode = "exponential"):

    def _exp(x,a,C1,C0):
    
        if x == 0:
            return C0 + C1
        else:
            return C1*np.exp( (-3*abs(x))/a )        

    # ====================================== #

    if mode == "exponential":
        _C = _exp

    C = []
    for j in dept:
        c0 = []
        for i in dept:
            #c0.append(gama_ajuste_gdg(j-i,VARIANCIA)) # Gutiérrez, Dvorkin e Grana (2014)
            c0.append(_C(j-i,a=a,C1=C1,C0=C0)) # Isaaks e Srivastava (1989)
        C.append(c0)

    return np.array(C)


def cov_matrix(M):

    COV = []
    for i in range(len(M)):
        _ = []
        for j in range(len(M)):
            _.append(np.cov(M[i],M[j])[0][1])
        COV.append(_)
    return np.array(COV)