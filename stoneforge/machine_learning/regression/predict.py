import numpy as np
import numpy.typing as npt
import pickle
import os

from sklearn.preprocessing import PolynomialFeatures
#from catboost.core import CatBoostRegressor

"linear_regression_simple",
"linear_regression_polynomial",
"decision_tree_regression",
"support_vector_regression",
"random_forest_regression",
'lightgbm_regression',
#'xgboost_regression',
#'catboost_regression'

def fit_load(path, method):
    full_path = os.path.join(path, method + "_fit_property.pkl")
    with open(full_path, 'rb') as f:
        return pickle.load(f)

def linear_regression(x: npt.ArrayLike, path, fit_info, **kwargs)-> np.ndarray:
    
    if path:
        method = 'linear_regression_simple'
        overall_fit_settings = fit_load(path, method)
    else:
        overall_fit_settings = pickle.loads(fit_info)

    pol_settings = overall_fit_settings['polinomial']
    slregression = overall_fit_settings['serialized_model']

    pol_degree = PolynomialFeatures(degree=pol_settings['degree'])
    x_poly = pol_degree.fit_transform(x)

    return slregression.predict(x_poly , **kwargs)


def support_vector_regression(x: npt.ArrayLike, path, fit_info, **kwargs) -> np.ndarray:

    
    if path:
        method = 'support_vector_regression'
        svnegression = fit_load(path, method)
    else:
        svnegression = pickle.loads(fit_info)

    if '2dy' in svnegression:
        y_pred = []
        for i in svnegression['2dy']:
            y_pred.append(svnegression['2dy'][i].predict(x, **kwargs))
        return np.array(y_pred).T
    else:
        return svnegression['1dy'].predict(x, **kwargs)


def decision_tree_regression(x: npt.ArrayLike, path, fit_info, **kwargs) -> np.ndarray:

    if path:
        method = 'decision_tree_regression'
        d_treec_regression = fit_load(path, method)
    else:
        d_treec_regression = pickle.loads(fit_info)

    return d_treec_regression.predict(x, **kwargs)


def random_florest_regression(x: npt.ArrayLike, path, fit_info, **kwargs) -> np.ndarray:

    if path:
        method = 'random_forest_regression'
        randomregression = fit_load(path, method)
    else:
        randomregression = pickle.loads(fit_info)
    
    return randomregression.predict(x, **kwargs)


# def xgboost_regression(x: npt.ArrayLike, path, fit_info, **kwargs)-> np.ndarray:
#
#   if path:
#       method = 'xgboost_regression'
#       xgboostregression = fit_load(path, method)
#   else:
#       xgboostregression = pickle.loads(fit_info)
#
#    return xgboostregression.predict(x,**kwargs)


def lightgbm_regression(x: npt.ArrayLike, path, fit_info, **kwargs)-> np.ndarray:
    
    if path:
        method = 'lightgbm_regression'
        lightregression = fit_load(path, method)
    else:
        lightregression = pickle.loads(fit_info)

    if '2dy' in lightregression:
        y_pred = []
        for i in lightregression['2dy']:
            y_pred.append(lightregression['2dy'][i].predict(x, **kwargs))
        return np.array(y_pred).T
    else:
        return lightregression['1dy'].predict(x, **kwargs)


# def catboost_regression(x: npt.ArrayLike, path, fit_info, **kwargs)-> np.ndarray:

#     if path:
#         method = 'catboost_regression'
#         catregression = fit_load(path, method)
#     else:
#         catregression = pickle.loads(fit_info)
    
#     if '2dy' in catregression:
#         y_pred = []
#         for i in catregression['2dy']:
#             y_pred.append(catregression['2dy'][i].predict(x, **kwargs))
#         return np.array(y_pred).T
#     else:
#         return catregression['1dy'].predict(x, **kwargs)



_predict_methods = {
    "linear_regression_simple": linear_regression,
    "linear_regression_polynomial": linear_regression,
    "support_vector_regression": support_vector_regression,
    "decision_tree_regression": decision_tree_regression,
    "random_forest_regression": random_florest_regression,
#    "xgboost_regression": xgboost_regression,
    "lightgbm_regression": lightgbm_regression,
#    "catboost_regression": catboost_regression,
    }


def predict(x: npt.ArrayLike, method: str = "linear_regression_simple", path = ".", fit_info = False,
              scalers=False, **kwargs):

    if method == "linear_regression_simple":
        fun = _predict_methods[method]
    if method == "linear_regression_polynomial":
        fun = _predict_methods[method]
    if method == "support_vector_regression":
        fun = _predict_methods[method]
    if method == "decision_tree_regression":
        fun = _predict_methods[method]
    if method == "random_forest_regression":
        fun = _predict_methods[method]
#    if method == "xgboost_regression":
#        fun = _predict_methods[method]
    if method == "lightgbm_regression":
        fun= _predict_methods[method]
#    if method == "catboost_regression":
#        fun= _predict_methods[method]
    if method == "scaler_regression":
        return 0

    if method != "scaler_regression":

        if not scalers:
            scaler = pickle.load(
                open(os.path.join(path,method+"_scaler.pkl"), 'rb'))
            scalerp = pickle.load(
                open(os.path.join(path,method+"_scalerp.pkl"), 'rb'))
        else:
            scaler,scalerp = scalers

        x_norm = pickle.loads(scaler).transform(x)
        x_norm = pickle.loads(scalerp).transform(x_norm)

        y = fun(x_norm, path, fit_info, **kwargs)
        return y